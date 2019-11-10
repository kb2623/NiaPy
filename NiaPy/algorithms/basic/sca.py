# encoding=utf8
from typing import Tuple, Callable, Dict, Union
import logging

import numpy as np

from NiaPy.util import Task
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic.SineCosineAlgorithm')
logger.setLevel('INFO')

__all__ = ['SineCosineAlgorithm']

class SineCosineAlgorithm(Algorithm):
	r"""Implementation of sine cosine algorithm.

	Algorithm:
		Sine Cosine Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705115005043

	Reference paper:
		Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022.

	Attributes:
		Name (List[str]): List of string representing algorithm names.
		a (float): Parameter for control in :math:`r_1` value
		Rmin (float): Minimu value for :math:`r_3` value
		Rmax (float): Maximum value for :math:`r_3` value

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['SineCosineAlgorithm', 'SCA']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of algorithm.

		Returns:
			Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022."""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[float, int]], bool ]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* a: TODO
			* Rmin: TODO
			* Rmax: TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'a': lambda x: isinstance(x, (float, int)) and x > 0,
			'Rmin': lambda x: isinstance(x, (float, int)),
			'Rmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def setParameters(self, NP: int = 25, a: float = 3, Rmin: float = 0, Rmax: float = 2, **ukwargs: dict) -> None:
		r"""Set the arguments of an algorithm.

		Args:
			NP: Number of individual in population
			a: Parameter for control in :math:`r_1` value
			Rmin: Minimu value for :math:`r_3` value
			Rmax: Maximum value for :math:`r_3` value

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.a, self.Rmin, self.Rmax = a, Rmin, Rmax

	def nextPos(self, x: np.ndarray, x_b: np.ndarray, r1: float, r2: float, r3: float, r4: float, task: Task) -> np.ndarray:
		r"""Move individual to new position in search space.

		Args:
			x: Individual represented with components.
			x_b: Best individual represented with components.
			r1: Number dependent on algorithm iteration/generations.
			r2: Random number in range of 0 and 2 * PI.
			r3: Random number in range [Rmin, Rmax].
			r4: Random number in range [0, 1].
			task: Optimization task.

		Returns:
			New individual that is moved based on individual ``x``.
		"""
		return task.repair(x + r1 * (np.sin(r2) if r4 < 0.5 else np.cos(r2)) * np.fabs(r3 * x_b - x), self.Rand)

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, dict]:
		r"""Initialize the individuals.

		Args:
			task: Optimization task

		Returns:
			1. Initialized population of individuals
			2. Function/fitness values for individuals
			3. Additional arguments
		"""
		return Algorithm.initPopulation(self, task)

	def runIteration(self, task: Task, P: np.ndarray, P_f: np.ndarray, xb: np.ndarray, fxb: float, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of Sine Cosine Algorithm.

		Args:
			task: Optimization task.
			P: Current population individuals.
			P_f: Current population individulas function/fitness values.
			xb: Current best solution to optimization task.
			fxb: Current best function/fitness value.
			dparams: Additional parameters.

		Returns:
			1. New population.
			2. New populations fitness/function values.
			3. New global best position.
			4. New global best position function/fitness value.
			5. Additional arguments.
		"""
		r1, r2, r3, r4 = self.a - task.Iters * (self.a / task.Iters), self.uniform(0, 2 * np.pi), self.uniform(self.Rmin, self.Rmax), self.rand()
		for i in range(len(P)):
			P[i] = self.nextPos(P[i], xb, r1, r2, r3, r4, task)
			P_f[i] = task.eval(P[i])
			if P_f[i] < fxb: xb, fxb = P[i].copy(), P_f[i]
		return P, P_f, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
