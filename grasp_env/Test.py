from stable_baselines3 import SAC
from stable_baselines3.sac.policies import CnnPolicy
from Environment import GraspEnv


env = GraspEnv()

model = SAC(CnnPolicy,
			env,
			buffer_size = 50000,
			verbose = 1,
			learning_starts = 2000,
			train_freq = (1, "episode"),
			seed = 1234)

number = 1

while True:

	model.learn(total_timesteps = (1e+5),
				log_interval = 1)

	model.save("SAC_model_" + str(number) + ".pt")

	number += 1