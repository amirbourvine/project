import gymnasium as gym
from gymnasium_env.envs.ElectricityMarket import ElectricityMarketEnv
import numpy as np
import pandas as pd
from config import env_type, N_FEATURES

def gen_seqs(env, num_seq, horizon):
    # generate the sequences of ds and ps

    seq_p = []
    seq_d = []

    for _ in range(num_seq):
        # obs is soc,d,p

        for _ in range(N_FEATURES):
            seq_p.append(0)
            seq_d.append(0)

        obs, info = env.reset()

        seq_p.append(obs[2])
        seq_d.append(obs[1])

        for _ in range(horizon-1):
            obs, reward, terminated, truncated, info = env.step(np.array([0]))
            # print(observation, reward, terminated, truncated, info)

            seq_p.append(obs[2])
            seq_d.append(obs[1])

            if terminated or truncated:
                break

    
    return seq_p, seq_d

def gen_dataset(env, num_seq, horizon):
    # saves sequences to csv file, aka dataset
    
    seq_p, seq_d = gen_seqs(env, num_seq, horizon)

    df_d = pd.DataFrame(seq_d, columns=['V6'])
    df_p = pd.DataFrame(seq_p, columns=['V6'])

    df_d.to_csv(f'env_{env_type}_d_nfeatures={N_FEATURES}.csv', index=False)
    df_p.to_csv(f'env_{env_type}_p_nfeatures={N_FEATURES}.csv', index=False)

    print("DONE!")




env = gym.make('gymnasium_env/ElectricityMarketEnv-v0', capacity=10 , max_t=5000)

# seq_p, seq_d = gen_seqs(env, 2, 10)
# print(seq_p)
# print(seq_d)

gen_dataset(env, 2, 5000)

env.close()