import gymnasium as gym
from forecast_agent import forecast_agent
from gymnasium_env.envs.ElectricityMarket import ElectricityMarketEnv


# logic to train and evaluate forecasting agent

env = gym.make('gymnasium_env/ElectricityMarketEnv-v0', capacity=10 , max_t=5000)

agent = forecast_agent()

agent.eval_agent(env, 5000)

env.close()