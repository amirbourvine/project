import gymnasium as gym
from stable_baselines3 import SAC, PPO
from gymnasium_env.envs.ElectricityMarket import ElectricityMarketEnv
import numpy as np
import os
from config import LOG_DIR, env_type

def train(env, model_type):
    model = None

    if model_type=="SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
        if env_type==2 or env_type==3 or env_type==4: 
            total_timesteps = 8_000
        if env_type==1: 
            total_timesteps = 3_000
        
    if model_type=="PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
        total_timesteps = 20_000

        if env_type==3: 
            total_timesteps = 100_000

        if env_type==4: 
            total_timesteps = 200_000


    if model == None:
        print("Error: no such model type!")
        return None

    os.makedirs(LOG_DIR, exist_ok=True)

    model.learn(total_timesteps=total_timesteps, tb_log_name=f"training_run_{model_type}_envType={env_type}")

    return model

