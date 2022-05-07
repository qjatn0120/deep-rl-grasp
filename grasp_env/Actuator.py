import numpy as np
from math import cos, sin
from scipy.spatial.transform import Rotation

class Actuator:

	def __init__(self, config, robot):

		self._robot = robot

		self._robot_config = config["robot"]

		self._max_translation = self._robot_config.get("max_translation", 0.03)

		self._max_yaw_rotation = self._robot_config.get("max_yaw_rotation", 0.15)

	def reset(self):

		self._robot.reset()

		self._robot.open_gripper()

		self._gripper_open = True

	def step(self, action):

		translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])

		_, ori = self._robot.get_pos()

		yaw = -Rotation.from_quat(ori).as_euler("xyz", degrees = False)[2]

		rot = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])

		translation[:2] = np.dot(translation[:2], rot)

		open_close = action[4]

		if open_close > 0 and not self._gripper_open:
			self._robot.open_gripper()
			self._gripper_open = True
		elif open_close < 0 and self._gripper_open:
			self._robot.close_gripper()
			self._gripper_open = False
		else:
			self._robot.move_robot(translation, yaw_rotation)

	def is_lift(self):
		return self._robot.get_gripper_distance() < 0.099 and not self._gripper_open

	def get_pos(self):
		return self._robot.get_pos()

	def _clip_translation_vector(self, translation, yaw):
		length = np.linalg.norm(translation)
		if length > self._max_translation:
			translation *= self._max_translation / length
		if abs(yaw) > self._max_yaw_rotation:
			yaw *= self._max_yaw_rotation / abs(yaw)
		return translation, yaw