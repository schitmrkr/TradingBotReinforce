import numpy as np
import random
import time
import copy

from DeepQDuelDouble import Agent
from utils import plot_learning_curve

file_to_open = "15m-w-ta-feature.csv"

data = np.loadtxt(file_to_open,
                  delimiter=",", dtype=float)



class TradeData():
    def __init__(self, data=data, window_size=100):
        self.raw_data = data
        self.window_size = window_size
        self.raw_mapping = {
                            "open_time": 0,
                            "open":1,
                            "high":2,
                            "low":3,
                            "close":4,
                            "volume":5,
                            "close_time":6,
                            "quote_volume":7,
                            "count":8,
                            "taker_buy_volume":9,
                            "taker_buy_quote_volume":10,
                            "ignore":11,
                            "rsi": 12,
                            "macd": 13,
                            "cci": 14, 
                            "adx": 15
                            }

    def sample_random_data_raw(self):
        random_index = np.random.randint(low=self.window_size, high=self.raw_data.shape[0])
        sample = self.raw_data[random_index-self.window_size: random_index]
        return sample

    def sample_data_raw(self, index):
        if index < self.window_size:
            return
        sample = self.raw_data[index-self.window_size: index]
        return sample
        
    def get_raw_state_from_sample(self, sample):
        raw_state = {
                    "close_time": [sample[:,self.raw_mapping["close_time"]]],
                    "close": [sample[:,self.raw_mapping["close"]]],
                    "volume": [sample[:,self.raw_mapping["volume"]]],
                    "rsi": [sample[:,self.raw_mapping["rsi"]]],
                    "macd": [sample[:,self.raw_mapping["macd"]]],
                    "cci": [sample[:,self.raw_mapping["cci"]]],
                    "adx": [sample[:,self.raw_mapping["adx"]]]
                }
        #print(sample.shape)
        return raw_state



class TradingEnv():
    def __init__(self, balance, data=data):
        self.trade_data = TradeData(data=data)
        self.initial_index = self.trade_data.window_size

        self.action_dims = 3
        self.action_space = [i for i in range(self.action_dims)]
        
        self.raw_state = self.trade_data.get_raw_state_from_sample(self.trade_data.sample_data_raw(self.initial_index))
        self.initial_balance = balance

        self.trade_amount = 100

        self.done = False

        """
        Buy Status Object
        {
            bought_for: 100,
            crypto_price: 26000,
            amount_crypto: bought_at / crypto_price,
        }
        """

        self.state = {
            "current_index": [self.initial_index],
            "reward": [0],
            "balance": [self.initial_balance],
            "buy_count": [0],
            "buy_details": [],
            "raw_state": self.raw_state,
        }

        self.action_map = {
            "buy":0, 
            "hold":1, 
            "sell":2
        }

    def reset(self):
        self.state = {
            "current_index": [self.initial_index],
            "reward": [0],
            "balance": [self.initial_balance],
            "buy_count": [0],
            "buy_details": [],
            "raw_state": self.raw_state,
        }
        return self.state
        

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already done. Please reset the environment.")

        if self.state["balance"][-1] < self.trade_amount and action == 0:
            raise Exception("Cannot buy because of low balance")

        if self.state["buy_count"][-1] < 1 and action == 2:
            raise Exception("Cannot sell because of nothing has been bought yet")

        old_state = copy.deepcopy(self.state)

        if action == 0:     #buy
            if self.state["balance"][-1] > self.trade_amount:
                self.state["raw_state"] = self.trade_data.get_raw_state_from_sample(self.trade_data.sample_data_raw(self.state["current_index"][-1]+1))
                self.state["reward"] = [0]
                self.state["balance"][-1] -= self.trade_amount
                self.state["buy_count"][-1] += 1

                buy_detail = {
                    "timestamp": self.raw_state["close_time"][-1][-1],
                    "bought_for": self.trade_amount,
                    "crypto_price": self.raw_state["close"][-1][-1],
                    "amount_crypto": self.trade_amount / self.raw_state["close"][-1][-1],
                }

                self.state["buy_details"].append(buy_detail)
        elif action == 1:   #hold
            self.state["raw_state"] = self.trade_data.get_raw_state_from_sample(self.trade_data.sample_data_raw(self.state["current_index"][-1]+1))
            self.state["reward"] = [0]
        elif action == 2:   #sell
            if self.state["buy_count"][-1] > 0:
                total_profit = 0
                total_used_balance = 0
                sold_price = self.state["raw_state"]["close"][-1][-1]
                for buy_detail in self.state["buy_details"]:
                    total_profit += ((sold_price - buy_detail["crypto_price"]) / buy_detail["crypto_price"]) * buy_detail["bought_for"]
                    total_used_balance += buy_detail["bought_for"]
            self.state["raw_state"] = self.trade_data.get_raw_state_from_sample(self.trade_data.sample_data_raw(self.state["current_index"][-1]+1))
            self.state["reward"] = [total_profit]
            self.state["balance"][-1] += total_profit + total_used_balance
            self.state["buy_count"][-1] = 0
            self.state["buy_details"] = []

        self.state["current_index"][-1] += 1

        #print(len(self.trade_data.raw_data))
        if self.state["current_index"][-1] >= len(self.trade_data.raw_data) - 1:
            self.done = True

        if self.state["buy_count"][-1] == 0 and self.state["balance"][-1] < self.trade_amount:
            self.done = True

        new_state = copy.deepcopy(self.state)

        #done for Q value calculation
        if action == 2:
            done = True
        else:
            done = False


        return old_state, new_state, self.state["reward"][-1], done

        

if __name__ == "__main__":
    env = TradingEnv(balance = 5000)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4, input_dims=100, n_actions=3, mem_size=10000, 
                    eps_min=0.01, batch_size=64, eps_dec=1e-3, replace=100)

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    filename = "agent-trader.png"
    num_episodes = 1000

    scores, eps_history = [], []

    for i in range(num_episodes):
        env.reset()
        init_state = env.state
        state = init_state
        score = 0
        i = 0
        while not env.done:
            action = agent.choose_action(state)
            #print(action)
            old_state, new_state, reward, done =  env.step(action)
            score += reward

            agent.store_transition(old_state, action, reward, new_state, int(done))
            agent.learn()

            state = new_state

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score, 
                'epsilon %.3f' % agent.epsilon)

        if i>10 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_episodes)]
    plot_learning_curve(x, scores, eps_history, filename)