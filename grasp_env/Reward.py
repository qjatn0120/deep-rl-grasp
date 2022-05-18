import numpy as np

class Reward():

	def __init__(self, config, actuator, world):

		self._world = world

		self._actuator = actuator

		self._reward_config = config["reward"]

		self._grip_reward = self._reward_config.get("grip_reward", 1)

		self._lift_reward = self._reward_config.get("lift_reward", 1)

		self._task_reward = self._reward_config.get("task_reward", 1)

		self._approach_reward = self._reward_config.get("approach_reward", 0.5)

	def reset(self):

		self._is_grip = False

		self._total_approach_reward = 0

		self._total_lift_reward = 0

		self._finish_task = False

		pos, _ = self._actuator.get_pos()

		self._init_object_distance = self._world.get_distance(pos)

	def get_reward(self):

		self._is_lift = self._actuator.is_lift()

		return self._get_approach_reward() + self._get_grip_reward() + self._get_lift_reward() + self._get_task_reward()

	def _get_approach_reward(self):

		pos, _ = self._actuator.get_pos()

		object_distance = self._world.get_distance(pos)

		if self._is_lift:
			r = self._approach_reward
		else:
			r = (self._init_object_distance - object_distance) * self._approach_reward * 2

		r = r - self._total_approach_reward

		self._total_approach_reward += r

		return r

	def _get_lift_pos(self):

		pos, _ = self._actuator.get_pos()
		return pos[2]

	def _get_grip_reward(self):

		if not self._is_grip and self._is_lift:
			self._is_grip = True
			self._lift_pos = self._get_lift_pos()
			return self._grip_reward

		if self._is_grip and not self._is_lift:
			self._is_grip = False
			r = 0 if self._finish_task else -self._grip_reward
			self._finish_task = False
			return r

		return 0

	def _get_lift_reward(self):

		if self._is_lift:
			r = (self._get_lift_pos() - self._lift_pos) * self._lift_reward
			self._lift_pos = self._get_lift_pos()
		else:
			r = -self._total_lift_reward

		self._total_lift_reward += r

		return r

	def _get_task_reward(self):

		r = self._world.remove_high_object()

		if r:
			self._total_lift_reward = 0
			self._finish_task = True
		return r * self._task_reward

def main():

	from World import World
	from Robot import Robot
	from Actuator import Actuator
	from Camera import Camera

	import os
	import yaml
	import pybullet_data
	conf_path = os.getcwd()
	conf_path = os.path.join(conf_path, "..", "conf.yaml")
	with open(conf_path, "r") as f:
		config = yaml.safe_load(f)

	robot_path = os.getcwd()
	robot_path = os.path.join(robot_path, "..", "model", "gripper.sdf")

	world = World(config)
	robot = Robot(robot_path, world)
	actuator = Actuator(config, robot)
	camera = Camera(config, robot)
	reward = Reward(config, actuator, world)

	import keyboard
	from time import sleep

	for n_epi in range(5):

		world.reset_sim()
		actuator.reset()
		reward.reset()

		grasp = 1.0

		total_reward = 0

		while True:

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

			actuator.step(action)

			camera.render_image()

			total_reward += reward.get_reward()

			print("Reward: ", total_reward)

			sleep(0.2)

if __name__ == "__main__":
	main()