"""
Modified from github repo: shakedzy/tic_tac_toe
"""
import numpy as np
import torch
import torch.nn as nn
import os
import time 
import random
from collections import deque 
import pickle

def init_weights(net):
    for m in net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

class TicTacToeNet(nn.Module):
    def __init__(self, n_actions=16):
        super(TicTacToeNet, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(512, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, n_actions)
        init_weights(self)
    
    def forward(self, x):
        # mask = x[:,1].reshape(x.shape[0],-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = nn.functional.softmax(x)
        # x = x * mask
        return x

class ReplayMemory():
    def __init__(self, size, seed=None):
        super(ReplayMemory, self).__init__()
        if seed:
            random.seed(seed)
        self._memory = deque(maxlen=size)
        self._counter = 0
    
    def __len__(self):
        return len(self._memory)

    def append(self, ele):
        self._memory.append(ele)
        self._counter += 1
    
    def counter(self):
        return self._counter 
    
    def sample(self, n):
        return random.sample(self._memory, n)

class DQN():
    def __init__(self, memory, gamma=0.95, batch_size=100, n_actions=16, lr=0.0001, device="cpu", learning_procedures_to_q_target_switch=1000):
        self.memory = memory
        self.net = TicTacToeNet(n_actions=n_actions)
        # self.q_net = TicTacToeNet(n_actions=n_actions)
        self.batch_size = batch_size
        self.gamma = gamma
        self.criterion = nn.MSELoss().to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.n_actions = n_actions
        self.learning_procedures_to_q_target_switch = learning_procedures_to_q_target_switch

    def _fetch_from_batch(self, batch, key):
        return np.array(list(map(lambda x: x[key], batch)))

    def learn(self):
        """state: [N, C, H, W]"""
        if self.memory.counter() % self.batch_size !=0 or self.memory.counter() == 0:
            # print("Passing on learning procedure")
            pass
        else:
            # print(len(self.memory))
            self.net.train()
            data = self.memory.sample(self.batch_size)
            # for i in data:
            #     if i['reward']!= 0:
            #         print(i)
            next_state = torch.tensor(self._fetch_from_batch(data, 'next_state')).type(torch.float32).to(self.device)
            ends = self._fetch_from_batch(data, 'is_end')
            reward = torch.tensor(self._fetch_from_batch(data, 'reward')).type(torch.float32).to(self.device)
            state = torch.tensor(self._fetch_from_batch(data, 'state')).type(torch.float32).to(self.device)
            action = self._fetch_from_batch(data, 'action')
            q_t = self.net(next_state)
            for i in range(ends.size):
                if ends[i]:
                    q_t[i] = torch.zeros(self.n_actions)
            future_q, _ = torch.max(q_t, dim=1)
            labels = reward + self.gamma * future_q
            self.optimizer.zero_grad()
            c_t = self.net(state)
            preds = c_t[torch.arange(c_t.size(0)), action]
            cost = self.criterion(preds, labels.detach())
            cost.backward()
            self.optimizer.step()
            # if self.memory.counter() % (self.learning_procedures_to_q_target_switch) == 0:
            #     # copy weights from q-net to q-target
            #     self.q_net.load_state_dict(self.net.state_dict())
            # print("Batch: %s | Q-Net cost: %s | Learning rate: %s", self.memory.counter//self.batch_size, cost, lr)
            return cost

    def act(self, state, epsilon, train=True):
        """
        state: [C, H, W]
        """
        rnd = random.random()
        if rnd < epsilon:
            # print(np.where(np.ravel(state[1]) > 0))
            empty = np.where(np.ravel(state[1]) > 0)[0]
            action = random.choice(empty)
        else: 
            state = torch.tensor(state).type(torch.float32).unsqueeze(0)
            self.net.eval()
            preds = self.net(state)
            if not train:
                mask = state[0][1].flatten()
                _, action = torch.max(preds*mask+mask*1e3, dim=1) # get the maximum value in the empty region
            else:
                _, action = torch.max(preds, dim=1)
            action = action.numpy()[0]
        return action

    def add_to_memory(self, state, action, reward, next_state, is_end):
        self.memory.append({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'is_end': is_end})

def index2coord(idx):
    # convert idx to coordinates
    return (idx//4, idx%4)

def coord2index(coord):
    # convert coordinates to index
    return coord[0] * 4 + coord[1]

class Game():
    def __init__(self, p1, p2, win_reward=1, lose_reward=-1, tie_reward=0, invalid_move_r=-10, blocks=4):
        super(Game, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.block_layer = 0
        self.empty_layer = 1
        self.p1_layer = 2
        self.p2_layer = 3
        self.win_r = win_reward
        self.lose_r = lose_reward
        self.tie_r = tie_reward
        self.blocks = blocks
        self.invalid_move_r = invalid_move_r
        self.reset()
    
    def reset(self):
        self.board = np.zeros((4,4,4))
        self.current_player = 2
        self._invalid_move_played = False
        self.random_block()

    def random_block(self):
        # total_blocks = np.size(self.board[self.block_layer])
        # block_ids = random.sample(list(total_blocks), self.blocks)
        xx, yy = np.meshgrid(list(range(4)), list(range(4)))
        board_index = list(zip(xx.ravel(), yy.ravel()))
        block_indices = random.sample(board_index, self.blocks)
        self.board[self.empty_layer] = 1
        for idx in block_indices:
            self.board[self.block_layer][idx] = 1
            self.board[self.empty_layer][idx] = 0
    
    def active_player(self):
        if self.current_player == 2:
            return self.p1
        else:
            return self.p2

    def inactive_player(self):
        if self.current_player == 2:
            return self.p2 
        else:
            return self.p1
    
    def play(self, cell):
        # if cell is integer, convert to [x,y]
        # if isinstance(cell, int):
        #     coord = index2coord(cell)
        # else:
        #     coord = cell
        coord = index2coord(cell)
        self._invalid_move_played = False
        if self.board[self.empty_layer][coord] == 0:
            self._invalid_move_played = True
            return {"win_status": None, "is_end": False, "invalid_move": True}
        self.board[self.current_player][coord] = 1
        self.board[self.empty_layer][coord] = 0
        status = self.game_status()
        return {"win_status": status['win_status'], 
                "is_end": status["is_end"], "invalid_move": False}
    
    def next_player(self):
        if not self._invalid_move_played:
            if self.current_player == 2:
                self.current_player = 3
            elif self.current_player == 3:
                self.current_player = 2
            else:
                raise("Unknown player")

    def game_status(self):
        def chain_length(x):
            longest = 0 
            counter = 0
            for i  in x:
                if i==1:
                    counter += 1
                    if counter > longest:
                        longest = counter
                else:
                    counter = 0
            return longest

        # compute score 
        scores = []
        empty_blocks = np.sum(self.board[self.empty_layer])
        # not terminal position or game over
        if empty_blocks:
            return {'is_end': False, "win_status": None}
        # terminal position for player 1
        # if empty_blocks == 1:
        #     cur_board = self.board.copy()
        #     last_pos = self.final_position()
        #     coord = index2coord(last_pos)
        #     cur_board[self.p2_layer][coord] = 1
        # else:
        #     cur_board = self.board.copy()
        cur_board = self.board.copy()
        for i in [self.p1_layer, self.p2_layer]:
            p_board = cur_board[i]
            v_chain = np.apply_along_axis(chain_length, 0, p_board)
            h_chain = np.apply_along_axis(chain_length, 1, p_board)
            p_score = np.sum((v_chain[v_chain>1]-1)**2) + np.sum((h_chain[h_chain>1]-1)**2)
            scores.append(p_score)
        if scores[0] > scores[1]:
            # player 1 is current player
            if self.current_player == 2:
                win_status = "win"
            else:
                win_status = "lose"
        elif scores[0] < scores[1]:
            # palyer 2 is current player
            if self.current_player == 3:
                win_status = "win"
            else:
                win_status = "lose"
        else:
            win_status = "tie"
        # if last position for player 1, game continues
        # if empty_blocks == 1:
        #     is_end = False
        # else:
        #     is_end = True
        return {'is_end': True, 'win_status': win_status}
    
    def final_position(self):
        empty_blocks = np.sum(self.board[self.empty_layer])
        if empty_blocks == 1:
            coords = np.where(np.ravel(self.board[self.empty_layer]) > 0)[0] 
            action = coords[0]
            return action

    def print_board(self):
        # row = ' '
        # status = self.game_status()
        # merge 4 layers
        vis = self.board[self.block_layer].astype("object")
        vis[vis==1] = "#"
        vis[self.board[self.p1_layer] == 1] = "x"
        vis[self.board[self.p2_layer] == 1] = "o"
        vis[self.board[self.empty_layer] == 1] = " "
        # print(vis)
        # print(self.board)
        print("---------")
        for i in vis:
            print("|"+"|".join(i)+"|")
            print("---------")
            
# test board map
# board=np.array([[[0,0,0,0],[1,0,0,0],[1,0,0,1],[0,1,0,0]],
# [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
# [[0,0,1,0],[0,0,1,1],[0,1,1,0],[0,0,0,1]],
# [[1,0,0,1],[0,1,0,0],[0,0,0,0],[1,0,1,0]],
# ])
class Player:
    """
    Base class for all player types
    """
    def __init__(self, name):
        self.name = name

    def shutdown(self):
        pass

    def add_to_memory(self, data):
        pass

    def save(self, filename):
        pass

    def select_cell(self, board, **kwargs):
        pass

    def learn(self, **kwargs):
        pass

class HumanPlayer(Player):
    def select_cell(self, board, player_id=1, **kwargs):
        cell = input("Select cell to fill: \n|0|1|2|3|\n|4|5|6|7|\n|8|9|10|11|\n|12|13|14|15|\ncell number:")
        # coord = index2coord(cell)
        # if player_id == 1:
        #     current_player = 2 
        # else:current_player = 3
        # board[current_player][coord] = 1
        # board[1][coord] = 0 
        # return board
        return int(cell)

class RandomPlayer(Player):
    def select_cell(self, board, **kwargs):
        empty = np.where(np.ravel(board[1]) > 0)[0]
        action = random.choice(empty)
        return action

class QPlayer(Player):
    def __init__(self, name, gamma=0.95, batch_size=100, memory_size=100000, lr=0.0001, device="cpu"):
        memory = ReplayMemory(size=memory_size)
        self.model = DQN(memory=memory, gamma=gamma, batch_size=batch_size, device=device, lr=lr)
        self.device = device
        self.name = name

    def select_cell(self, board, epsilon, train=True):
        return self.model.act(board, epsilon, train=train)

    def learn(self):
        return self.model.learn()

    def add_to_memory(self, data):
        # def _swap(x):
        #     temp = x[-1]
        #     x[-1], x[-2] = x[-2], temp
        #     return x
        state = data["state"]
        next_state = data["next_state"]
        # if player_id == 2: 
        #     state = _swap(state)
        #     next_state = _swap(next_state)
        self.model.add_to_memory(state=state, next_state=next_state, action=data["action"],
                                 reward=data["reward"], is_end=data["is_end"])

    def save(self, filename):
        torch.save(self.model.net.state_dict(), filename)
    
    def load(self, filename):
        state_dict=torch.load(filename, map_location=torch.device(self.device))
        self.model.net.load_state_dict(state_dict)
    
def moveX(blocked, Xmoves, Ymoves):
    pass

def train(batch_size=200, gamma=0.95, lr=0.0001, memory_size=int(1e5), num_games=int(1e6), save_dir="test", device="cpu", n_actions=16):
    # epsilon: annealing propability to use random action
    # max [0.6, 0.1, 0.05], min: [0.01] for games [<0.25, <0.5, >0.5]
    # set random seed
    seed = int(time.time() * 1000)
    random.seed(seed)
    p1 = QPlayer(name="1P", lr=lr, gamma=gamma, batch_size=batch_size, memory_size=memory_size, device=device)
    # p2 = QPlayer(name="2P", lr=lr, gamma=gamma, batch_size=batch_size, memory_size=memory_size, device=device)
    p2 = RandomPlayer(name="2P")
    # statistics 
    total_rewards = {p1.name: 0, p2.name:0}
    costs = {p1.name:[], p2.name:[]}
    rewards = {p1.name:[], p2.name:[]}
    start_t = time.time()
    best_reward = 0
    for g in range(1,num_games+1):
        game = Game(p1, p2) if g%2!=0 else Game(p2, p1)  # both players play x and o
        # game = Game(p1, p2)
        last_phases = {p1.name: None, p2.name: None} # store last state 
        # for debug
        # player_list = []
        # action_list = {p1.name:[], p2.name:[]}

        # while True:
        #     if isinstance(game.active_player(), HumanPlayer):
        #         game.print_board()
        #         print("{}'s turn".format(game.active_player().name))
        #     state = np.copy(game.board)
        #     # action_list[game.active_player().name].append(action)
        #     if last_phases[game.active_player().name] is not None:
        #         last_phases[game.active_player().name]['next_state'] = state
        #         is_end = play_status["is_end"]
        #         last_phases[game.active_player().name]['is_end'] = is_end
        #         win_status = play_status['win_status']
        #         # if play_status["invalid_move"]:
        #         #     r = game.invalid_move_r
        #         if win_status is not None:
        #             # previous player status
        #             if win_status == "win":
        #                 r = game.lose_r
        #             elif win_status == "lose":
        #                 r = game.win_r
        #             else:
        #                 r = game.tie_r
        #         else:
        #             r = 0
        #         last_phases[game.active_player().name]["reward"] = r
        #         # if last_phases[game.active_player().name]['reward'] != 0:
        #         #     print(last_phases[game.active_player().name], game.active_player().name, win_status, g)
        #         # if g > 13:
        #         #     exit()
        #         game.active_player().add_to_memory(last_phases[game.active_player().name])
        #         # print(len(p1.model.memory), g)
        #         if is_end:
        #             # print(last_phases)
        #             break
        #     # compute annealed epsilon
        #     if g <= num_games // 4:
        #         max_eps = 0.6
        #     elif g <= num_games // 2:
        #         max_eps = 0.1 
        #     else:
        #         max_eps = 0.05
        #     min_eps = 0.01 
        #     eps = round(max(max_eps-round(g*(max_eps-min_eps)/num_games, 3), min_eps), 3)
        #     # play and receive reward
        #     action = game.active_player().select_cell(state, epsilon=eps)
        #     last_phases[game.active_player().name] = {'state': state, 'action': action}
        #     # play action
        #     play_status = game.play(action)
        #     # learning with loss 
        #     cost = game.active_player().learn()
        #     if cost is not None:
        #         costs[game.active_player().name].append(cost)
        #     # if not play_status['is_end']:
        #     # player_list.append(game.active_player().name)
        #     game.next_player()
        while True:
            if isinstance(game.active_player(), HumanPlayer):
                game.print_board()
                print("{}'s turn".format(game.active_player().name))
            state = np.copy(game.board)
            # action_list[game.active_player().name].append(action)
            if last_phases[game.active_player().name] is not None:
                data = last_phases[game.active_player().name]
                data['next_state'] = state
                data['is_end'] = is_end
                game.active_player().add_to_memory(data)
                if is_end:
                    break
            # compute annealed epsilon
            if g <= num_games // 4:
                max_eps = 0.6
            elif g <= num_games // 2:
                max_eps = 0.1 
            else:
                max_eps = 0.05
            min_eps = 0.01 
            eps = round(max(max_eps-round(g*(max_eps-min_eps)/num_games, 3), min_eps), 3)
            # play and receive reward
            action = game.active_player().select_cell(state, epsilon=eps)
            # play action
            play_status = game.play(action)
            win_status = play_status['win_status']
            # if play_status["invalid_move"]:
            #     r = game.invalid_move_r
            is_end = play_status['is_end']
            if play_status["invalid_move"]:
                r = game.invalid_move_r
            elif is_end:
                if win_status == "win":
                    r = game.win_r
                elif win_status == "lose":
                    r = game.lose_r
                else:
                    r = game.tie_r
                # print(win_status, r)
            else:
                r = 0
            last_phases[game.active_player().name] = {'state': state, 'action': action, 'reward': r}
            total_rewards[game.active_player().name] += r
            if r == game.win_r:
                total_rewards[game.inactive_player().name] += game.lose_r 
            elif r == game.lose_r:
                total_rewards[game.inactive_player().name] += game.win_r
            # if last_phases[game.active_player().name]['reward'] != 0:
            #     print(last_phases[game.active_player().name], game.active_player().name, win_status, g)
            # if g > 4:
            #     exit()
            # game.active_player().add_to_memory(last_phases[game.active_player().name])
            # learning with loss 
            cost = game.active_player().learn()
            if cost is not None:
                costs[game.active_player().name].append(cost)
            if not is_end:
                game.next_player()

        # add last phase for active player
        # data = last_phases[game.active_player().name]
        # data["next_state"] = np.zeros((4,4,4))
        # data["is_end"] = True
        # game.active_player().add_to_memory(data)
        data = last_phases[game.inactive_player().name]
        data["next_state"] = np.zeros((4,4,4))
        data["is_end"] = True
        if r == game.win_r:
            data['reward'] = game.lose_r 
        elif r == game.lose_r:
            data['reward'] = game.win_r
        game.inactive_player().add_to_memory(data)

        # total_rewards[game.active_player().name] += r
        # if r == game.win_r:
        #     total_rewards[game.inactive_player().name] += game.lose_r
        # elif r == game.lose_r:
        #     total_rewards[game.inactive_player().name] += game.win_r

        # print(last_phases, game.active_player().name)

        # print(player_list, action_list)

        # print statistics
        if g%100 == 0:
            print('Game:{} | Number of training {}/{} | Epsilon: {} | Average Reward: {}/{} vs {}/{} | cost: {}'.format(
                g, len(costs[p1.name]), len(costs[p2.name]), eps, total_rewards[p1.name]/100, p1.name, total_rewards[p2.name]/100, p2.name, costs[p1.name][-1]
            ))
            rewards[p1.name].append(total_rewards[p1.name]/100)
            rewards[p2.name].append(total_rewards[p2.name]/100)
            total_rewards = {p1.name:0, p2.name: 0}
        # save best reward model
        if total_rewards[p1.name] > best_reward:
            best_model = os.path.join(save_dir, "best_model.pt")
            p1.save(best_model)
            best_reward = total_rewards[p1.name]
        # save model weights periodically
        if g % 10000 == 0:
            p1_save_model = os.path.join(save_dir, "{}_epoch_{}.pt".format(p1.name, g))
            # p2_save_model = os.path.join(save_dir, "{}_epoch_{}.pt".format(p2.name, g))
            p1.save(p1_save_model)
            # p2.save(p2_save_model)

    # trainin time
    finish_time = time.time() - start_t
    print("Training took {}m:{}s".format(finish_time//60, finish_time%60))
    with open(os.path.join(save_dir, "reward.pkl"), "wb") as f:
        pickle.dump(rewards, f)
    f.close()
    with open(os.path.join(save_dir, "cost.pkl"), "wb") as f:
        pickle.dump(costs, f)
    f.close()
    print("best reward:", best_reward)

def play(model_path, machine_first=False, player_type="human"):
    if machine_first:
        p1 = QPlayer(name="1P")
        p1.load(model_path)
        if player_type == "human":
            p2 = HumanPlayer(name="2P")
        elif player_type == "random":
            p2 = RandomPlayer(name="2P")
    else:
        if player_type == "human":
            p1 = HumanPlayer(name="1P")
        elif player_type == "random":
            p1 = RandomPlayer(name="1P")
        p2 = QPlayer(name="2P")
    print("Game start!")
    game = Game(p1, p2)
    while not game.game_status()['is_end']:
        if isinstance(game.active_player(), HumanPlayer):
            game.print_board()
        state = np.copy(game.board)
        action = game.active_player().select_cell(state, epsilon=0, train=False)
        # print(action)
        play_status = game.play(action)
        if not play_status['is_end']:
            game.next_player()
    print("game over")
    print(game.game_status())
    print("==============")
    print(game.print_board())
    print("==============")
        

if __name__ == "__main__":
    # x = torch.randn(10,4,4,4)
    # model = TicTacToeNet(16)
    # res = model(x)
    # print(res.shape)
    train()
    # play("best_model.pt")
    # play("test/1P_epoch_1000000.pt", player_type="random", machine_first=True)

