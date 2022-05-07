import numpy as np
import pybullet_data
import os
from scipy.spatial.transform import Rotation

class BaseScene():

	def __init__(self, world, config, gen, train):
		"""
		@param world: given world to generate scene 
		@type: World
		"""

		self._world = world
		self._objects_conf = config["objects"]
		self._world_conf = config["world"]
		self._floor_conf = config["floor"]
		self._gen = gen
		self._train = train
		self._model_path = pybullet_data.getDataPath()
		self._min_objects = self._objects_conf.get("min_objects", 1)
		self._max_objects = self._objects_conf.get("max_objects", 6)
		self._extent = self._objects_conf.get("extent", 0.2)
		self._spawn_z = self._objects_conf.get("spawn_z", 0.1)
		self._spawn_delay = self._objects_conf.get("spawn_delay", 0.2)
		self._reset_delay = self._world_conf.get("reset_delay", 1.)

	def _sample_random_objects(self, n_objects):
		"""
		@param n_objects: number of objects to sample
		@type: int

		@return path of selected objects
		@rtype: [str] * n_obnjects
		"""

		if self._train:
			object_range = np.arange(800, 1000)
		else:
			object_range = 800

		selection = self._gen.choice(object_range, size = n_objects)
		paths = [os.path.join(self._model_path, "random_urdfs",
			"{0:03d}/{0:03d}.urdf".format(i)) for i in selection]

		return paths

	def get_targets(self):
		return self._targets

	def remove_target(self, object_id):

		self._targets.remove(object_id)

class FloorScene(BaseScene):

	def reset(self):
		self._plane_path = os.path.join(self._model_path, "plane.urdf")
		self._targets = []
		start_pos, start_ori = self._get_start_pos_and_ori()
		self._world.add_object(
			path = self._plane_path,
			start_pos = start_pos,
			start_ori = start_ori)
		n_objects = self._gen.integers(low = self._min_objects, high = self._max_objects + 1, size = 1).item()
		paths = self._sample_random_objects(n_objects)

		for path in paths:
			position = np.r_[self._gen.uniform(low = -self._extent, high = self._extent, size = 2), 0.1]
			orientation = Rotation.random(num = 1, random_state = self._gen).as_quat().squeeze()
			object_id = self._world.add_object(path, position, orientation)
			self._targets.append(object_id)
			self._world.run(self._spawn_delay)

		self._world.run(self._reset_delay)

	def _get_start_pos_and_ori(self):
		start_pos_x = self._floor_conf.get("start_pos_x", 0)
		start_pos_y = self._floor_conf.get("start_pos_y", 0)
		start_pos_z = self._floor_conf.get("start_pos_z", 0)
		start_pos = [start_pos_x, start_pos_y, start_pos_z]
		start_ori_x = self._floor_conf.get("start_ori_x", 0)
		start_ori_y = self._floor_conf.get("start_ori_y", 0)
		start_ori_z = self._floor_conf.get("start_ori_z", 0)
		start_ori_w = self._floor_conf.get("start_ori_w", 1)
		start_ori = [start_ori_x, start_ori_y, start_ori_z, start_ori_w]
		return start_pos, start_ori