import gymnasium as gym
from gymnasium_env.envs.ElectricityMarket import ElectricityMarketEnv
import numpy as np

def main():
    env = gym.make('gymnasium_env/ElectricityMarketEnv-v0', capacity=10 , max_t=5000)

    s_size = env.observation_space.shape[0]
    a_size = env.action_space

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

    print(env.reset())

    print("***************** ACTIONS *****************")

    # observation, reward, terminated, truncated, info = env.step(np.array([1.00]))
    # print(observation, reward, terminated, truncated, info)

    # observation, reward, terminated, truncated, info = env.step(np.array([-1.00]))
    # print(observation, reward, terminated, truncated, info)

    for i in range(810):
        observation, reward, terminated, truncated, info = env.step(np.array([0]))

        if i>=800:
            print(observation, reward, terminated, truncated, info)

    

    # obs is (soc, d_t, p_t)

if __name__ == "__main__":
    main()
