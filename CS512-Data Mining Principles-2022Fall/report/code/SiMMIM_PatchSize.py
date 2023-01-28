#Import common dependencies
import torch
import pandas as pd
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from PIL import Image 
from skimage import io
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import random
import os
from sklearn.metrics import roc_auc_score
import copy
from timm.models.layers import trunc_normal_
from SwinTransformer.models.swin_transformer_v2 import SwinTransformerV2
from SwinTransformer.utils import get_grad_norm
from timm.scheduler.cosine_lr import CosineLRScheduler



if torch.cuda.is_available:
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
    
def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    return None
setup_seed(0)




train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.Pad((0,120), padding_mode='constant'))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.RandomVerticalFlip())
train_transform.transforms.append(transforms.RandomResizedCrop(384, scale=(0.5, 1.1), ratio=(1.0,1.0)))
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))

test_transform = transforms.Compose([])
test_transform.transforms.append(transforms.Pad((0,120), padding_mode='constant'))
test_transform.transforms.append(transforms.RandomResizedCrop(384, scale=(1.0, 1.0), ratio=(1.0,1.0)))
test_transform.transforms.append(transforms.ToTensor())
test_transform.transforms.append(transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))




class MaskGenerator:
    def __init__(self, input_size=384, mask_patch_size=32, model_patch_size=4, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    
class hist_dataset_with_mask(Dataset):
    def __init__(self, df_path, train = False, mask_config=None):
        self.df = pd.read_csv(df_path)
        self.train = train
        self.mask_generator = MaskGenerator(mask_config["input_size"], mask_config["mask_patch_size"], mask_config["model_patch_size"], mask_config["mask_ratio"])
        self.set_mask = MaskGenerator()()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image_path = self.df.iloc[idx]['FilePath']
        image = Image.open(image_path)
        
        if self.train:
            image_tensor = train_transform(image)
        else:
            image_tensor = test_transform(image)

        label = self.df.loc[idx]['Label']
        label = torch.tensor(label, dtype=torch.long)
        
        if self.train:
            mask = self.mask_generator()
        else:
            mask = self.set_mask
        
        return image_tensor, mask, label





class SwinTransformerForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride=32):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}





def train_fun(model, train_loader, val_loader,file_name='SiMMIM_32ps'):
    EPOCHS = 500
    WARMUP_EPOCHS = 50
    n_iter_per_epoch = len(train_loader)
    num_steps = int(EPOCHS * n_iter_per_epoch)
    warmup_steps = int(WARMUP_EPOCHS * n_iter_per_epoch)
    multi_steps = []
    optimizer = optim.AdamW(simmim.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=5e-4, weight_decay=0.05)
    lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=5e-6,
                warmup_lr_init=5e-7,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )

    start_time = time.time()
    
    train_loss_return = []
    grad_norm_return = []
    val_loss_return = []
    best_val_loss = np.inf

    if os.path.exists(file_name+'.txt'):
        with open(file_name+'.txt', "r+") as f:
            f.seek(0)
            f.truncate()
            
    for epoch in range(EPOCHS):
        # Training steps
        model.train()
        loss_every_epoch = []
        grad_norm_every_epoch = []
        for idx, (img, mask, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            mask = mask.to(device)
            loss = model(img, mask)
            # print(loss)
            loss.backward()
            grad_norm = get_grad_norm(model.parameters())
            #print(grad_norm)
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
            loss_every_epoch.append(loss.item())
            grad_norm_every_epoch.append(grad_norm)
        train_loss_return.append(np.mean(loss_every_epoch))
        grad_norm_return.append(np.mean(grad_norm_every_epoch))
        
        print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,EPOCHS))
        print('Train set | Loss: {:6.4f} | grad_norm: {:6.4f}'\
              .format(np.mean(loss_every_epoch), np.mean(grad_norm_every_epoch)))
        
        with open(file_name+'.txt','a') as file0:
            print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,EPOCHS),file=file0)
            print('Train set | Loss: {:6.4f} | grad_norm: {:6.4f}'\
                  .format(np.mean(loss_every_epoch),np.mean(grad_norm_every_epoch)),file=file0)
                  
        model.eval()
        loss_every_epoch = []
        with torch.no_grad():
            for idx, (img, mask, _) in enumerate(val_loader):
                img = img.to(device)
                mask = mask.to(device)
                loss = model(img, mask)
                loss_every_epoch.append(loss.item())
            val_loss = np.mean(loss_every_epoch)
            val_loss_return.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        save_model(model, train_loss_return, val_loss_return, grad_norm_return, best_model_wts, file_name=file_name) 
                  
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        print('Test  set | Loss: {:6.4f} | Best Val Loss: {:6.4f} | time elapse: {:>9}'\
              .format(val_loss, best_val_loss, elapse))
        
        with open(file_name+'.txt','a') as file0:
            print('Test  set | Loss: {:6.4f} | Best Val Loss: {:6.4f} | time elapse: {:>9}'\
                  .format(val_loss, best_val_loss, elapse),file=file0)

    return None

def save_model(model, train_loss_return, val_loss_return, grad_norm_return, best_model_wts, file_name):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'val_loss':val_loss_return, 'grad_norm':grad_norm_return}
    torch.save(state, file_name+'.pt')
    return None



if __name__ == "__main__":
    train_df_path = 'train.csv'
    val_df_path = 'val.csv'
    test_df_path = 'test.csv'
    BS = 1
    num_workers = 4
    mask_config = {"input_size": 384, 
                "mask_patch_size": 32, 
                "model_patch_size":4, 
                "mask_ratio": 0.5}
    train_loader = DataLoader(hist_dataset_with_mask(train_df_path, train=True, mask_config=mask_config),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(hist_dataset_with_mask(val_df_path, train=False, mask_config=mask_config),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(hist_dataset_with_mask(test_df_path, train=False, mask_config=mask_config),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=False)

    model = SwinTransformerForSimMIM(img_size=384,
                          patch_size=4,
                          in_chans=3,
                          num_classes=0,
                          embed_dim=128,
                          depths=[2, 2, 18, 2],
                          num_heads=[ 4, 8, 16, 32],
                          window_size=24,
                          mlp_ratio=4.,
                          qkv_bias=True,
                          drop_rate=0.0,
                          drop_path_rate=0.2,
                          ape=False,
                          patch_norm=True,
                          use_checkpoint=False,
                          pretrained_window_sizes=[ 12, 12, 12, 6 ])


    # pretrained_state = {}
    # weight_path = 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
    # weight = torch.load(weight_path)['model']
    # model_state = model.state_dict()
    # for key in model_state.keys():
    #     if key in weight.keys():
    #         pretrained_state[key] = weight[key]
    #     else:
    #         pretrained_state[key] = model_state[key]
    #         print(key)
    # model.load_state_dict(pretrained_state)

    simmim = SimMIM(encoder=model)

    simmim = simmim.to(device)
    train_fun(simmim, train_loader, val_loader)

