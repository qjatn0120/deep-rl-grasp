from stable_baselines3 import SAC
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from Environment import GraspEnv

env = GraspEnv()
model = SAC(CnnPolicy, env, buffer_size = 50000, verbose = 1)
model.learn(total_timesteps = (1e+5), log_interval = 1)