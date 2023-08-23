import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.input_shape = input_shape
        self.mem_cntr = 0

        self.balance_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_balance_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.buy_count_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_buy_count_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.close_price_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_close_price_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.volume_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_volume_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.rsi_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_rsi_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.macd_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_macd_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.cci_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_cci_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.adx_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_adx_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)

        #self.state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        #self.new_state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.balance_memory[index] = state['balance'][-1]
        self.new_balance_memory[index] = state_['balance'][-1]
        self.buy_count_memory[index] = state['buy_count'][-1]
        self.new_buy_count_memory[index] = state_['buy_count'][-1]


        self.close_price_memory[index] = state['raw_state']['close'][-1]
        self.new_close_price_memory[index] = state_['raw_state']['close'][-1]
        self.volume_memory[index] = state['raw_state']['volume'][-1]
        self.new_volume_memory[index] = state_['raw_state']['volume'][-1]
        self.rsi_memory[index] = state['raw_state']['rsi'][-1]
        self.new_rsi_memory[index] = state_['raw_state']['rsi'][-1]
        self.macd_memory[index] = state['raw_state']['macd'][-1]
        self.new_macd_memory[index] = state_['raw_state']['macd'][-1]
        self.cci_memory[index] = state['raw_state']['cci'][-1]
        self.new_cci_memory[index] = state_['raw_state']['cci'][-1]
        self.adx_memory[index] = state['raw_state']['adx'][-1]
        self.new_adx_memory[index] = state_['raw_state']['adx'][-1]

        #self.state_memory[index] = state
        #self.new_state_memory[index] = state_

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        #print("here")
        #print(batch)

        states = {
            "current_index": None,
            "reward":None,
            "balance":self.balance_memory[batch],
            "buy_count":self.buy_count_memory[batch],
            "buy_details": None,
            "raw_state": {
                "close_time": None,
                "close": self.close_price_memory[batch],
                "volume": self.volume_memory[batch],
                "rsi": self.rsi_memory[batch],
                "macd": self.macd_memory[batch],
                "cci": self.cci_memory[batch],
                "adx": self.adx_memory[batch]
            }
        }

        states_ = {
            "current_index": None,
            "reward":None,
            "balance":self.new_balance_memory[batch],
            "buy_count":self.new_buy_count_memory[batch],
            "buy_details": None,
            "raw_state": {
                "close_time": None,
                "close": self.new_close_price_memory[batch],
                "volume": self.new_volume_memory[batch],
                "rsi": self.new_rsi_memory[batch],
                "macd": self.new_macd_memory[batch],
                "cci": self.new_cci_memory[batch],
                "adx": self.new_adx_memory[batch]
            }
        }
        
        #states = self.state_memory[batch]
        #states_ = self.new_state_memory[batch]

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir + name)

        self.input_dim = input_dim
        self.n_actions = n_actions

        #Neural Divergence
        self.layernorm_price = nn.LayerNorm(input_dim)
        self.conv1d_price = nn.Conv1d(1, 8, 10)
        self.conv1d_price_2 = nn.Conv1d(8,1,10)

        self.conv1d_macdXrsi = nn.Conv1d(1,8,10)
        self.conv1d_adxXcci = nn.Conv1d(1, 8, 10)

        self.conv1d_macdXrsi_2 =  nn.Conv1d(8,1,10)
        self.conv1d_adxXcci_2 =  nn.Conv1d(8,1,10)

        self.layernorm_volume = nn.LayerNorm(input_dim)
        self.layernorm_rsi = nn.LayerNorm(input_dim)
        self.layernorm_macd = nn.LayerNorm(input_dim)
        self.layernorm_cci = nn.LayerNorm(input_dim)
        self.layernorm_adx = nn.LayerNorm(input_dim)

        self.fc1_volume = nn.Linear(input_dim, 512)
        self.fc1_features = nn.Linear(input_dim-18, 512)
        self.fc1_price = nn.Linear(input_dim-18, 512)

        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        close_price = torch.FloatTensor(np.array([state["raw_state"]["close"]]))
        volume = torch.FloatTensor(np.array([state["raw_state"]["volume"]]))
        rsi = torch.FloatTensor(np.array([state["raw_state"]["rsi"]]))
        macd = torch.FloatTensor(np.array([state["raw_state"]["macd"]]))
        cci = torch.FloatTensor(np.array([state["raw_state"]["cci"]]))
        adx = torch.FloatTensor(np.array([state["raw_state"]["adx"]]))

        #print(close_price.shape)

        close_price = close_price.view(-1,1,self.input_dim)
        volume = volume.view(-1,1,self.input_dim)
        rsi = rsi.view(-1,1,self.input_dim)
        macd = macd.view(-1,1,self.input_dim)
        cci = cci.view(-1,1,self.input_dim)
        adx = adx.view(-1,1,self.input_dim)

        #print(close_price.shape)

        price_norm = self.layernorm_price(close_price)
        price_conv = self.conv1d_price(price_norm)
        price_final = self.conv1d_price_2(price_conv)

        volume_norm = self.layernorm_volume(volume)

        rsi_norm = self.layernorm_rsi(rsi)
        macd_norm = self.layernorm_macd(macd)
        cci_norm = self.layernorm_cci(cci)
        adx_norm = self.layernorm_adx(adx)

        rsi_conv = self.conv1d_macdXrsi(rsi_norm)
        macd_conv = self.conv1d_macdXrsi(macd_norm)

        rsi_macd_div = rsi_conv + macd_conv
        rsi_macd_conv = self.conv1d_macdXrsi_2(rsi_macd_div)

        cci_conv = self.conv1d_adxXcci(cci_norm)
        adx_conv = self.conv1d_adxXcci(adx_norm)

        cci_adx_div = cci_conv + adx_conv
        cci_adx_conv = self.conv1d_adxXcci_2(cci_adx_div)

        conv_last_layer = cci_adx_conv + rsi_macd_conv

        last_price = self.fc1_price(price_final)
        last_volume = self.fc1_volume(volume_norm)
        last_feature = self.fc1_features(conv_last_layer)

        last_layer_to_process = last_price + last_volume + last_feature

        V = self.V(last_layer_to_process)
        A = self.A(last_layer_to_process)

        #print(A.shape)
        #print(V.shape)

        A = A.view(-1, self.n_actions)
        V = V.view(-1, 1)

        #print(A.shape)
        #print(V.shape)

        #print(state["buy_count"])

        if state["buy_count"][-1] == 0:
            A_new = A.clone()
            A_new[:,2] = torch.min(A)
            A = A_new
        elif state["buy_count"][-1] >= 5:
            A_new = A.clone()
            A_new[:,0] = torch.min(A)
            A = A_new
        else:
            pass

        return V, A
        
    def save_checkpoints(self):
        print('.....saving checkpoint.....')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoints(self):
        print('.....loading checkpoint......')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                    mem_size, batch_size, eps_min=0.01, eps_dec=1e-5,
                    replace=1000, chkpt_dir='tmp/duelingddqn/'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.chkpt_dir = chkpt_dir

        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions, input_dim=input_dims, 
                                            name='btc-15m-ddqn-qeval',
                                            chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions, input_dim=input_dims, 
                                            name='btc-15m-ddqn-qnext',
                                            chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        #if np.random.random() > self.epsilon:
        if np.random.random() > 0:
            state = observation
            _, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else:
            if observation["buy_count"] == 0:
                action = np.random.choice([0,1])
            elif observation["buy_count"] > 0 and observation["buy_count"] < 5:
                action = np.random.choice([self.action_space])
            else:
                action = np.random.choice([1,2])

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                                        self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoints()
        self.q_next.save_checkpoints()

    def load_models(self):
        self.q_eval.load_checkpoints()
        self.q_next.load_checkpoints()

    def learn(self):
        torch.autograd.set_detect_anomaly(True)
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                            self.memory.sample_buffer(self.batch_size)

        states = state
        states_ = new_state

        actions = torch.tensor(action).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)
    
        q_next[dones.bool()] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        torch.autograd.set_detect_anomaly(False)
