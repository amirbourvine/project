from typing import Optional
import numpy as np
import gymnasium as gym
from config import env_type

class ElectricityMarketEnv(gym.Env):

    metadata = {"render_modes": [], "render_fps": 0}

    def demand(self, t:int) -> float:

        if env_type==3:
            t = float(t)/4.0

        if env_type==4:
            t = float(t)/4000.0
        
        
        arg0 = -1*((float(t)-0.4)**2)
        arg1 = 2*((0.05)**2)
        arg2 = 100*(np.exp(arg0/arg1))

        arg3 = -1*((float(t)-0.7)**2)
        arg4 = 2*((0.1)**2)
        arg5 = 120*(np.exp(arg3/arg4))

        val = arg2+arg5

        noise = np.random.normal(0, abs(val)/50)

        return val+noise



    def price(self, t:int) -> float:

        if env_type==3:
            t = float(t)/4.0

        if env_type==4:
            t = float(t)/3000.0

        arg0 = -1*((float(t)-0.5)**2)
        arg1 = 2*((0.15)**2)
        arg2 = 80*(np.exp(arg0/arg1))

        arg3 = -1*((float(t)-1.0)**2)
        arg4 = 3*((0.03)**2)
        arg5 = 140*(np.exp(arg3/arg4))

        val = arg2+arg5

        noise = np.random.normal(0, abs(val)/50)

        return val+noise

    def __init__(self, capacity: float, max_t: int, power_production_cost: float = 1.0, init_soc: float = 0.0):
        # The size of the square grid
        self.capacity = capacity
        self.max_t = max_t
        self.init_soc = init_soc
        self.power_production_cost = power_production_cost
        self.soc = -1 #will be initialized in reset, and updated in step
        self.d = -1 #will be initialized in reset, and updated in step
        self.p = -1 #will be initialized in reset, and updated in step
        self.t = -1 #will be initialized in reset, and updated in step

        # Observations are (soc, d_t, p_t)
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0]), high=np.array([self.capacity,np.inf,np.inf]), dtype=np.float64)

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float64)
    
    def _get_obs(self):
        return np.array([self.soc,self.d, self.p], dtype=np.float64)
    
    # def get_capacity(self):
    #     return self.capacity

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.t = 0
        self.d = self.demand(t=0)
        self.p = self.price(t=0)
        self.soc = self.init_soc

        observation = self._get_obs()

        return observation, {"capacity": self.capacity}
    
    def step(self, action):
        # assuming action is numpy array, as defined in action_space
        old_soc = self.soc
        self.soc = np.minimum(self.capacity, np.maximum(self.soc+action[0]*self.capacity, 0))
        self.t = self.t + 1

        terminated = (self.t == self.max_t)
        truncated = False

        reward = 0
        info = {}

        if self.soc > old_soc:
            if env_type==2:
                reward = -(self.power_production_cost*(self.soc-old_soc))
            if env_type==1 or env_type==3 or env_type==4:
                reward = 0
        if self.soc-old_soc < -self.d:
            reward = ((old_soc-self.soc)-self.d)*self.p
            info = {"demand": (self.d), "total": (old_soc-self.soc)}
        
        self.d = self.demand(t=self.t)
        self.p = self.price(t=self.t)

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info