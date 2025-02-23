import gymnasium as gym
from train_model import train
from eval_model import eval

env = gym.make('gymnasium_env/ElectricityMarketEnv-v0', capacity=10 , max_t=5000)

model = train(env, model_type="SAC")

vec_env = model.get_env()
obs = vec_env.reset()
print(f"obs: {obs}")

for i in range(810):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    if i >= 800:
        print(f"action: {action}")
        print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")


profit_sum, demand_per = eval(model, 5000)

env.close()