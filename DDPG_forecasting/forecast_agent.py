from train_forecast import train_forecast_agent
import numpy as np

import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config import N_FEATURES

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class forecast_agent:
    def __init__(self):
        print("start: train D")
        tf.reset_default_graph()  # Important: Reset the graph before creating new agents

        # with tf.variable_scope("agent_d"):
        self.forecast_d, self.d_state_scaler, self.A_d,self.B_d = train_forecast_agent(is_d=True)
        print("end: train D")
        print("start: train P")
        # with tf.variable_scope("agent_p"):
        self.forecast_p, self.p_state_scaler,  self.A_p,self.B_p = train_forecast_agent(is_d=False)
        print("end: train P")
        self.must_sell = False

    def predict(self, curr_battery, state_d, state_p, capacity, file=None):
        if curr_battery==0.0:
            return np.array([1.0])
        
        if self.must_sell:
            self.must_sell = False
            return np.array([-1.0])


        if state_d[-1] >= capacity:
            return np.array([0.0])

        sell_now = state_p[-1]*max((capacity-state_d[-1]),0)

        new_state_p = (self.p_state_scaler.transform(state_p.reshape(1,-1)))[0]
        new_state_d = (self.d_state_scaler.transform(state_d.reshape(1,-1)))[0]

        next_p = self.forecast_p.choose_action(new_state_p)[0]
        next_p = next_p*(self.A_p-self.B_p)+self.B_p
        next_p = max(next_p,0)

        next_d = self.forecast_d.choose_action(new_state_d)[0]
        next_d = next_d*(self.A_d-self.B_d)+self.B_d
        next_d = max(next_d,0)



        if not (file is None):
            file.write(f"{next_p=}\n")
            file.write(f"{next_d=}\n")

        sell_next = next_p*max((capacity-next_d),0)

        if sell_now > sell_next:
            return np.array([-1.0])
        else:
            self.must_sell = True
            return np.array([0.0])



    def eval_agent(self, env, horizon):

        file = open("forecast_eval_log.txt", "w")

        obs, info = env.reset()

        capacity = info["capacity"]
        file.write(f"{capacity=}\n")

        profit_sum = 0.0
        total = 0.0
        demand = 0.0

        state_d = np.zeros(shape=(N_FEATURES))
        state_d[-1] = obs[1]

        state_p = np.zeros(shape=(N_FEATURES))
        state_p[-1] = obs[2]

        file.write(f"{obs=}\n")

        for _ in range(horizon):
            action = self.predict(obs[0], state_d, state_p, capacity, file)

            file.write(f"{action=}\n")

            obs, reward, terminated, truncated, info = env.step(action)
            
            file.write(f"{obs=} || {reward=}\n")

            # print(f"{reward=}")
            # print(f"{info=}")

            # [self.soc,self.d, self.p]

            if reward > 0:
                profit_sum += reward

            if "demand" in info:
                total += info["total"]
                demand += info["demand"]

            if terminated or truncated:
                break

            state_d = np.roll(state_d, -1)
            state_d[-1] = obs[1]

            state_p = np.roll(state_p, -1)
            state_p[-1] = obs[2]
        
        val = 0
        if total>0:
            val = ((demand/total)*100)

        print(f"profit_sum: {profit_sum}, demand_per: {val}")

        file.close()

        return profit_sum, val