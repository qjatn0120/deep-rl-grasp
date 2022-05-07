import pybullet as p
import numpy as np
import time
from scipy.spatial.transform import Rotation

class Robot():

	def __init__(self, description_path, world):
		"""
		@param description_path: directory of robot model file
		@type: str

		@param world: world to place robot
		@type: World
		"""

		self._world = world

		self._uid = self._world.get_uid()

		self._robot_path = description_path

	def reset(self):

		# TODO: parameterize pos and ori of robot

		robot_id = 	self._world.add_object(
						path = self._robot_path,
						start_pos = [0, 0, 0.6],
						start_ori = [1, 0, 0, 0]
					)

		self._id = robot_id

		self._all_joints = np.array(range(
			p.getNumJoints(self._id, physicsClientId = self._uid)))

		self._movable_joints = self.get_movable_joints()

		self._nq = len(self._movable_joints)

		self._all_joint_names = self.get_joint_names()

		self._x_axis_joint = self.get_joint_by_name("x_axis")

		self._y_axis_joint = self.get_joint_by_name("y_axis")

		self._z_axis_joint = self.get_joint_by_name("z_axis")

		self._rotate_joint = self.get_joint_by_name("base_joint")

		self._left_finger = self.get_joint_by_name("gripper_left_hinge_joint")

		self._right_finger = self.get_joint_by_name("gripper_right_hinge_joint")

	def get_movable_joints(self):
		"""
		@return: ids of all the movable joints
		@rtype: np.ndarray(shape: (self._nq))
		"""

		movable_joints = []
		for i in self._all_joints:
			joint_info = p.getJointInfo(
				self._id, i, physicsClientId = self._uid)
			q_index = joint_info[3]
			if q_index > -1:
				movable_joints.append(i)
		return movable_joints

	def get_joint_names(self):
		"""
		@return: Names of all the joints
		@rtype: np.ndarray(shale: [string] * self._nq)
		"""

		joint_names = []
		for i in self._all_joints:
			joint_info = p.getJointInfo(
				self._id, i, physicsClientId = self._uid)
			name = joint_info[1]
			name = name.decode("utf-8")
			joint_names.append(name)
		return np.array(joint_names)

	def get_joint_by_name(self, joint_name):
		"""
		@param joint_name: name of joint
		@type: str

		@return: Joint id of given joint name
		@rtype: int
		"""

		if joint_name in self._all_joint_names:
			joint_index = np.where(self._all_joint_names == joint_name)[0]
			if len(joint_index) > 1:
				raise Exception("Duplicated joint name error")
			return joint_index[0]
		else:
			raise Exeption("Not existing joint name error")

	def _set_joint_position(self, joint_index, position, max_force = 100.):
		if not joint_index in self._movable_joints:
			raise Exception("try to move unmovable joint")
		p.setJointMotorControl2(
			bodyUniqueId = self._id,
			jointIndex = joint_index,
			controlMode = p.POSITION_CONTROL,
			targetPosition = position,
			force = max_force)

	def _get_joint_position(self, joint_index):
		pos, _, _, _ = p.getJointState(
			bodyUniqueId = self._id,
			jointIndex = joint_index,
			physicsClientId = self._uid)
		return pos

	def get_pos(self):
		pos, ori, _, _, _, _ = p.getLinkState(self._id, 3)
		return np.array(pos), np.array(ori)

	# TODO: parameterize run time for each step
	def move_robot(self, translation, yaw_rotation):
		pos, ori = self.get_pos()
		yaw = Rotation.from_quat(ori).as_euler("xyz", degrees = False)[2]
		pos += translation
		yaw += yaw_rotation

		self._set_joint_position(self._x_axis_joint, pos[0])

		self._set_joint_position(self._y_axis_joint, -pos[1])

		self._set_joint_position(self._z_axis_joint, -pos[2] + 0.6)

		self._set_joint_position(self._rotate_joint, -yaw)

		self._world.run(0.1)

	def close_gripper(self):
		self._set_joint_position(self._left_finger, 0.05)
		self._set_joint_position(self._right_finger, 0.05)
		self._world.run(0.2)

	def open_gripper(self):
		self._set_joint_position(self._left_finger, 0)
		self._set_joint_position(self._right_finger, 0)
		self._world.run(0.2)

	def get_gripper_distance(self):
		return self._get_joint_position(self._left_finger) + self._get_joint_position(self._right_finger)


def main():
	import os
	import yaml
	from World import World
	import pybullet_data
	import time
	conf_path = os.getcwd()
	conf_path = os.path.join(conf_path, "..", "conf.yaml")
	with open(conf_path, "r") as f:
		config = yaml.safe_load(f)

	# TODO: move gripper sdf file to local folder
	robot_path = pybullet_data.getDataPath()
	robot_path = os.path.join(robot_path, "gripper", "wsg50_one_motor_gripper_new_free_base.sdf")

	world = World(config)
	robot = Robot(robot_path, world)
	world.reset_sim()
	robot.reset()
	time.sleep(2)
	robot.move_robot([0.1, 0.1, 0], 0.1)
	time.sleep(5)

if __name__ == "__main__":
	main()