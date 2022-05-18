import os
import yaml
import gym
import numpy as np
from gym import spaces

from World import World
from Camera import Camera
from Robot import Robot
from Actuator import Actuator
from Reward import Reward

class GraspEnv(gym.Env):

	def __init__(self):

		conf_path = os.getcwd()
		conf_path = os.path.join(conf_path, "..", "conf.yaml")
		with open(conf_path, "r") as f:
			self._config = yaml.safe_load(f)

		robot_path = os.getcwd()
		robot_path = os.path.join(robot_path, "..", "model", "gripper.sdf")

		self._world = World(self._config)
		self._robot = Robot(robot_path, self._world)
		self._actuator = Actuator(self._config, self._robot)
		self._camera = Camera(self._config, self._robot)
		self._reward = Reward(self._config, self._actuator, self._world)

		self._max_translation = self._config["robot"]["max_translation"]
		self._max_yaw = self._config["robot"]["max_yaw_rotation"]

		self._use_rgb_image = self._config["camera"]["rgb"]
		self._use_depth_image = self._config["camera"]["depth"]

		self._image_row = self._config["camera"]["image_row"]
		self._image_col = self._config["camera"]["image_column"]

		self._max_step_per_episode = self._config["environment"]["max_step_per_episode"]
		self.num_envs = 1

		self._image_channel = 0;
		if self._use_rgb_image:
			self._image_channel += 3
		if self._use_depth_image:
			self._image_channel += 1

		self.observation_space = spaces.Box(low = 0, high = 255, shape = (self._image_channel, self._image_row, self._image_col), dtype = np.uint8)

		self.action_space = spaces.Box(
			low = np.array([-self._max_translation, -self._max_translation, -self._max_translation, -self._max_yaw, -1.0]),
			high = np.array([self._max_translation, self._max_translation, self._max_translation, self._max_yaw, 1.0]),
			dtype = np.float32,
		)

	def reset(self):
		self._world.reset_sim()
		self._actuator.reset()
		self._reward.reset()
		self._n_step = 0

		state = self._camera.render_image()

		return state

	def step(self, action):

		self._actuator.step(action)

		self._n_step += 1

		# Calculate State
		state = self._camera.render_image()

		# Calculate Reward
		reward = self._reward.get_reward()

		# Calculate Done mask
		done = self._world.is_done()

		# Calculate Info
		info = {}

		if self._n_step == self._max_step_per_episode:
			done = True
		
		return (state, reward, done, info)

# test works for depth image only
def main():

	import keyboard
	from time import sleep

	env = GraspEnv()

	for n_epi in range(5):

		env.reset()

		grasp = 1.0

		total_reward = 0

		done = False

		while not done:

			if keyboard.is_pressed("left arrow"):
					action = [-1, 0, 0, 0, grasp]
			elif keyboard.is_pressed("right arrow"):
				action = [1, 0, 0, 0, grasp]
			elif keyboard.is_pressed("up arrow"):
				action = [0, 1, 0, 0, grasp]
			elif keyboard.is_pressed("down arrow"):
				action = [0, -1, 0, 0, grasp]
			elif keyboard.is_pressed("a"):
				action = [0, 0, 1, 0, grasp]
			elif keyboard.is_pressed("d"):
				action = [0, 0, -1, 0, grasp]
			elif keyboard.is_pressed("q"):
				action = [0, 0, 0, -1, grasp]
			elif keyboard.is_pressed("e"):
				action = [0, 0, 0, 1, grasp]
			elif keyboard.is_pressed("space"):
				grasp = 1.0 if grasp < 0 else -1.0
				action = [0, 0, 0, 0, grasp]
			else:
				action = [0, 0, 0, 0, grasp]

			action = np.array(action)

			state, reward, done, info = env.step(action)

			total_reward += reward

			print("Reward:", total_reward)

			sleep(0.2)

		print("Finish Episode:", n_epi)

if __name__ == "__main__":
	main()