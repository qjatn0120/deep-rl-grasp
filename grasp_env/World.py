import pybullet as p
import numpy as np
import math
from Scene import FloorScene

from numpy.random import Generator, PCG64
from gym.utils import seeding

class World():

	def __init__(self, config = None):

		if config is None:
			raise Exception("config file not found error")

		if not "world" in config:
			raise Exception("no world config in config file")

		self._world_conf = config["world"]
		self._visualize = self._world_conf.get("visualize", True)
		self._train = self._world_conf.get("train", True)
		self._train_seed = self._world_conf.get("train_seed", None)
		if self._train_seed == "None":
			self._train_seed = None
		self._test_seed = self._world_conf.get("test_seed", 5678)
		if self._train:
			seed = self._train_seed
		else:
			seed = self._test_seed
		self._world_type = self._world_conf.get("world_type", "Floor World")
		self._reset_delay = self._world_conf.get("reset_delay", 1.)
		self._gen = self.seed(seed)
		self._time_step = 1. / self._world_conf.get("time_step", 240)
		self._solver_iteration = self._world_conf.get("solver_iteration", 150)
		self._lift_height = self._world_conf.get("lift_height", 0.5)
		self._objects = []

		self._uid = p.connect(
			p.GUI if self._visualize else p.DIRECT)

		print(self._visualize)

		if self._world_type == "Floor World":
			self._scene = FloorScene(self, config, self._gen, self._train)

	def seed(self, seed = None):
		"""
		@param seed: fixed seed for same sequence of objects
		@type: int

		@return: random generator for generating objects
		@rtype: Generator
		"""

		if seed is None:
			gen = np.random.default_rng()
		else:
			gen = Generator(PCG64(seed))
		return gen

	def add_object(self, path, start_pos, start_ori, scailing = 1.):
		"""
		@param path: path of object to spawn
		@type: str

		@param start_pos: init position of object
		@type: [float] * 3

		@param start_ori: init orientation of object
		@type: [float] * 4

		@param scaliing: scale of object:
		@type: float
		"""

		extension = path.split('.')[-1]
		if extension == "urdf":
			object_id = p.loadURDF(
				fileName = path,
				useFixedBase = False,
				physicsClientId = self._uid,
				basePosition = start_pos,
				baseOrientation = start_ori
			)
		elif extension == "sdf":
			object_id = p.loadSDF(
				sdfFileName = path,
				physicsClientId = self._uid
			)[0]
			p.resetBasePositionAndOrientation(
				bodyUniqueId = object_id,
				posObj = start_pos,
				ornObj = start_ori,
				physicsClientId = self._uid
			)
		else:
			raise Exception("Unknown model file extension")

		self._objects.append(object_id)

		return object_id

	def run(self, duration):
		for _ in range(int(duration / self._time_step)):
			self.step_sim()

	def step_sim(self):
		p.stepSimulation(
			physicsClientId = self._uid)

	def reset_sim(self):
		p.resetSimulation(
			physicsClientId = self._uid)
		p.setPhysicsEngineParameter(
			fixedTimeStep = self._time_step,
			numSolverIterations = self._solver_iteration,
			enableConeFriction = 1,
			physicsClientId = self._uid)
		p.setGravity(
			gravX = 0,
			gravY = 0,
			gravZ = -9.8,
			physicsClientId = self._uid)
		self._models = []
		self._scene.reset()

	def get_uid(self):
		return self._uid

	def remove_high_object(self):

		res = 0

		targets = self._scene.get_targets()

		for object_id in targets:

			pos, _ = p.getBasePositionAndOrientation(
				bodyUniqueId = object_id,
				physicsClientId = self._uid)

			pos = pos[2]

			if pos > self._lift_height:

				res += 1

				self._scene.remove_target(object_id)

				p.removeBody(
					bodyUniqueId = object_id,
					physicsClientId = self._uid)

		return res

	def is_done(self):

		return len(self._scene.get_targets()) == 0

	def get_distance(self, pos):

		distance = []

		for object_id in self._scene.get_targets():

			obj_pos, _ = p.getBasePositionAndOrientation(
				bodyUniqueId = object_id,
				physicsClientId = self._uid)

			distance.append(math.dist(pos, obj_pos))

		return min(distance)

			

def main():

	import os
	import yaml
	conf_path = os.getcwd()
	conf_path = os.path.join(conf_path, "..", "conf.yaml")
	with open(conf_path, "r") as f:
		config = yaml.safe_load(f)

	world = World(config)
	world.reset_sim()
	from time import sleep
	sleep(5)

if __name__ == "__main__":
	main()