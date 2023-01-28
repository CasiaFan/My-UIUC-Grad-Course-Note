import torch
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, sampler
import glob, os 
import pandas as pd 
import numpy as np 
import re
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import torch.optim as optim 
import copy 
import time
import fire
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve
from scipy import stats


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


    
class hist_dataset(Dataset):
    def __init__(self, df_path, train = False):
        self.df = pd.read_csv(df_path)
        self.train = train
        
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
        
        return image_tensor, label


class LogitResnet(nn.Module):
    """ResNet architecture for extracting feature. Add an extra fc layer for extracting embedding."""
    def __init__(self, model_name, num_classes, return_logit=False, use_pretrained=True):
        """embeding """
        super(LogitResnet, self).__init__()
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=use_pretrained)
        else:
            print("unknown resnet model")
            exit()
        num_features = model.fc.in_features
        self.return_logit = return_logit
        self.net = nn.Sequential(*list(model.children())[:-1])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, inputs):
        x = self.net(inputs)
        # x = self.avgpool(x)
        l = torch.flatten(x, 1)
        x = self.fc(l)
        if self.return_logit:
            return [x, l]
        return [x]


def train(model, 
          model_save_path, 
          dataloader, 
          datasize,
          optimizer, 
          criterion, 
          num_epochs, 
          device="cpu",
          eval_metric="acc",
          num_classes=2):
    """Train Classifier"""
    best_perf = 0
    train_hist = {}
    train_hist["C_loss"] = []
    train_hist["perf"] = []
    start_t = time.time() 
    # best model save path
    best_test_model = os.path.join(model_save_path, "best_model.pt") 
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 50)
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            epoch_preds, epoch_gts = [], []
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                if phase == "test":
                    epoch_preds += preds.cpu().detach().tolist()
                    epoch_gts += labels.cpu().detach().tolist()
            ds = datasize[phase]
            epoch_loss = running_loss / ds 
            if phase == "test":
                if eval_metric == "acc":
                    epoch_perf = accuracy_score(epoch_gts, epoch_preds)
                    print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_perf))
                elif eval_metric == "f1":
                    avg_type = "binary" if num_classes == 2 else "weighted"
                    epoch_perf = f1_score(epoch_gts, epoch_preds, average=avg_type)
                    print("{} Loss: {:.4f}, F1: {:.4f}".format(phase, epoch_loss, epoch_perf))
                train_hist["perf"].append(epoch_perf)
            else:
                train_hist["C_loss"].append(epoch_loss)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
            if phase == 'test' and epoch_perf > best_perf:
                best_perf = epoch_perf
                torch.save(model.state_dict(), best_test_model)
        # if not ((epoch+1) % 5):
        #     torch.save(model.state_dict(), model_save_path+"/w_epoch_{}.pt".format(epoch+1))
    time_elapsed = time.time() - start_t 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val performance: {:.4f}".format(best_perf))
    return model, train_hist


def save_model(model, train_loss_return, val_loss_return, grad_norm_return, best_model_wts, file_name):
    state = {'best_model_wts':best_model_wts, 'model':model, \
             'train_loss':train_loss_return, 'val_loss':val_loss_return, 'grad_norm':grad_norm_return}
    torch.save(state, file_name+'.pt')
    return None

# eval model performance
def calculate_auc(pred, label, target_class=0):
    # label = label_binarize(label, classes=list(range(num_classes)))
    label = np.eye(num_classes)[label]
    fpr, tpr, _ = roc_curve(label[:, target_class], pred[:,target_class])
    roc_auc = auc(fpr, tpr)
    return roc_auc

def predict(model_name, 
            num_classes, 
            model_weights, 
            dataloader,
            device="cpu",  
            auc=True):
    model = LogitResnet(model_name, num_classes)
    state_dict=torch.load(model_weights, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    # result matrics:
    #      __________________________________________
    #      | gt \ pred | Normal | COVID | Pneumonia | 
    #      ------------------------------------------
    #      |  Normal   |        |       |           |
    #      |  COVID    |        |       |           |
    #      |  Pneumonia|        |       |           |
    #      ------------------------------------------
    
    result_matrics = np.zeros((3, 3))
    with torch.no_grad():
        if auc:
            pred_list, label_list = None, []
        for img, label in dataloader:
            tag = label.numpy()[0]
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            _, pred = torch.max(outputs[0], 1)
            score = outputs[0].cpu().numpy()
            if auc:
                if pred_list is None:
                    pred_list = score
                else:
                    pred_list = np.concatenate([pred_list, score], axis=0)
                label_list.append(tag)
            pred = int(pred.item())
            result_matrics[tag][pred] += 1
        # comput auc 
        if auc:
            auc_v = calculate_auc(pred_list, label_list, target_class=1)
            print("AUC: ", auc_v)

    print("result matrics: ", result_matrics)
    # ACC: TP+TN / ALL
    res_acc = []
    # sensitivity: TP / (TP + FN)
    res_sens = []
    # specificity: TN / (TN+FP)
    res_speci = []
    # f1 score: 2TP/(2TP+FP+FN)
    f1_score = []
    tol = 1e-8
    for i in range(num_classes):
        TP = result_matrics[i,i]
        FN = np.sum(result_matrics[i,:])-TP
        spe_matrics = np.delete(result_matrics, i, 0)
        FP = np.sum(spe_matrics[:, i])
        TN = np.sum(spe_matrics) - FP
        acc = TP/(TP+FP+tol)
        sens = TP/(TP+FN+tol)
        speci = TN/(TN+FP+tol)
        f1 = 2*TP/(2*TP+FP+FN+tol)
        res_acc.append(acc)
        res_speci.append(speci)
        res_sens.append(sens)
        f1_score.append(f1)
    print('Precision: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_acc[0],res_acc[1],res_acc[2], np.mean(res_acc)))
    print('Sensitivity: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_sens[0],res_sens[1],res_sens[2], np.mean(res_sens)))
    print('Specificity: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_speci[0],res_speci[1],res_speci[2], np.mean(res_speci)))
    print('F1 score: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(f1_score[0],f1_score[1],f1_score[2], np.mean(f1_score)))    
    

if __name__ == "__main__":
    train_df_path = 'train.csv'
    val_df_path = 'val.csv'
    test_df_path = 'test.csv'
    BS = 1
    num_workers = 4
    train_loader = torch.utils.data.DataLoader(hist_dataset(train_df_path, train=True),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(hist_dataset(val_df_path, train=False),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(hist_dataset(test_df_path, train=False),\
                            batch_size=BS, num_workers=num_workers, pin_memory=True, shuffle=False)

    model_name="resnet18" 
    image_dir = ""
    image_size=384
    num_classes=2
    batch_size=8
    num_epochs=100
    model_save_path="test"
    device="cuda:0" # "cuda:0"
    lr=0.001
    moment=0.9
    use_pretrained=True
    eval_metric="acc"

    os.makedirs(model_save_path, exist_ok=True)

    # get classifier 
    train_file = "/Users/zongfan/Projects/spie/test/example/train1/train1.txt"
    test_file = "/Users/zongfan/Projects/spie/test/example/train1/val1.txt"

    config = {"image_size": image_size, "train": train_file, "test": test_file}
    dataloaders = {"train": train_loader, "test": val_loader}
    datasizes = {x: len(hist_dataset(train_df_path, train="train"==x)) for x in ["train", "test"]}
    # loss function
    cls_weight = [1.0, 4.0]
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cls_weight)).to(device) # classification loss NOTE: https://en.wikipedia.org/wiki/Cross_entropy

    # load model 
    model = LogitResnet("resnet50", num_classes)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=moment) # NOTE: SGD: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    model_fit, hist = train(model=model, 
                            model_save_path=model_save_path, 
                            dataloader=dataloaders, 
                            datasize=datasizes,
                            optimizer=optimizer, 
                            criterion=criterion, 
                            num_epochs=num_epochs, 
                            device=device,
                            num_classes=num_classes,
                            eval_metric=eval_metric)
    # torch.save(model_ft.state_dict(), model_save_path+'/best_model.pt') 
    for k, v in hist.items():
        print("{}: {}".format(k, v))


    model_weights="/Users/zongfan/Downloads/HE_cls/test/best_model.pt"

    predict(model_name, 
            num_classes, 
            model_weights, 
            test_loader,
            image_dir, 
            image_size, 
            device="cpu",  
            auc=True)

    