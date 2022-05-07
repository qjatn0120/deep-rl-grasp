import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation
from math import degrees

class Camera():

	def __init__(self, config, robot):

		self._camera_config = config["camera"]

		self._camera_dist = self._camera_config.get("distance", 0.5)

		self._near_val = self._camera_config.get("near_val", 0.1)

		self._far_val = self._camera_config.get("far_val", 1)

		self._rgb = self._camera_config.get("rgb", True)

		self._depth = self._camera_config.get("depth", True)

		self._image_row = self._camera_config.get("image_row", 256)

		self._image_column = self._camera_config.get("image_column", 256)

		self._robot = robot

	def render_image(self):

		robot_pos, robot_ori = self._robot.get_pos()

		yaw = Rotation.from_quat(robot_ori).as_euler("xyz", degrees = False)[2]

		view_matrix = p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition = robot_pos - np.array([0, 0, self._camera_dist + 0.08]),
			distance = self._camera_dist,
			yaw = degrees(yaw),
			pitch = -90,
			roll = 0,
			upAxisIndex = 2,
		)

		proj_matrix = p.computeProjectionMatrixFOV(
			fov = 60,
			aspect = 1,
			nearVal = self._near_val,
			farVal = self._far_val,
		)

		width, height, rgb_image, depth_image, seg_image = p.getCameraImage(
			width = self._image_row,
			height = self._image_column,
			viewMatrix = view_matrix,
			projectionMatrix = proj_matrix,
			renderer = p.ER_BULLET_HARDWARE_OPENGL
		)

		rgb_image = np.array(rgb_image[:, :,:3])
		rgb_image = np.transpose(rgb_image, (2, 0, 1))
		depth_image = np.array(depth_image * 255, dtype = np.uint8)
		depth_image = np.expand_dims(depth_image, axis = 0)
		
		if self._rgb and self._depth:
			return np.concatenate((rgb_image, depth_image), axis = 0)
		elif self._rgb:
			return rgb_image
		elif self._depth:
			return depth_image
		else:
			raise Exception("You should choose at least one camera image")

def main():

	from World import World
	from Robot import Robot
	from Actuator import Actuator

	import os
	import yaml
	import pybullet_data
	conf_path = os.getcwd()
	conf_path = os.path.join(conf_path, "..", "conf.yaml")
	with open(conf_path, "r") as f:
		config = yaml.safe_load(f)

	robot_path = pybullet_data.getDataPath()
	robot_path = os.path.join(robot_path, "gripper", "wsg50_one_motor_gripper_new_free_base.sdf")

	world = World(config)
	robot = Robot(robot_path, world)
	actuator = Actuator(config, robot)
	camera = Camera(config, robot)

	world.reset_sim()
	actuator.reset()
	camera.render_image()

	from time import sleep

	for i in range(10):
		actuator.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, -1.0, 0.0, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, 0.0, 0.05, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, 0.0, -0.05, 0.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(20):
		actuator.step(np.array([0.0, 0.0, 0.0, -1.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(10):
		actuator.step(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
		camera.render_image()
		sleep(0.01)

	for i in range(11):
		actuator.step(np.array([0.0, 0.0, 0.0, 0.0, -1.0 if i % 2 else 1.0]))
		camera.render_image()
		sleep(0.01)

	import matplotlib.pyplot as plt
	rgb_image, _ = camera.render_image()
	rgb_image = np.transpose(rgb_image, (1, 2, 0))
	plt.imshow(rgb_image)
	plt.show()

if __name__ == "__main__":
	main()