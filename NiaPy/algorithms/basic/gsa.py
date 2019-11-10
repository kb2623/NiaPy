# encoding=utf8
from typing import Tuple, Union, Callable, Dict, List
import logging

import numpy as np

from NiaPy.util import Task
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GravitationalSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class GravitationalSearchAlgorithm(Algorithm):
	r"""Implementation of gravitational search algorithm.

	Algorithm:
		Gravitational Search Algorithm

	Date:
		2018

	Author:
		Klemen Berkoivč

	License:
		MIT

	Reference URL:
		https://doi.org/10.1016/j.ins.2009.03.004

	Reference paper:
		Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name: List[str] = ['GravitationalSearchAlgorithm', 'GSA']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of GravitationalSearchAlgorithm algorithm.

		Returns:
			Basic information.
		"""
		return r"""Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255"""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[int, float]], bool]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* G_0 (Callable[[Union[int, float]], bool])
			* epsilon (Callable[[float], bool])

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'G_0': lambda x: isinstance(x, (int, float)) and x >= 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		})
		return d

	def setParameters(self, NP: int = 40, G_0: float = 2.467, epsilon: float = 1e-17, **ukwargs: dict) -> None:
		r"""Set the algorithm parameters.

		Arguments:
			G_0: Starting gravitational constant.
			epsilon: Small values.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.G_0, self.epsilon = G_0, epsilon

	def G(self, t: int) -> float:
		r"""TODO.

		Args:
			t: Algorithms iteration number.

		Returns:
			TODO
		"""
		return self.G_0 / t

	def d(self, x: np.ndarray, y: np.ndarray, ln: int = 2) -> float:
		r"""Get distance between to planets.

		Args:
			x: First planet position.
			y: Second planet position.
			ln: Factor of distance.

		Returns:
			Distance between two planets.
		"""
		return np.sum((x - y) ** ln) ** (1 / ln)

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, dict]:
		r"""Initialize staring population.

		Args:
			task: Optimization task.

		Returns:
			1. Initialized population.
			2. Initialized populations fitness/function values.
			3. Additional arguments:
				* v (numpy.ndarray[float]): Velocity of planets.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		X, X_f, _ = Algorithm.initPopulation(self, task)
		v = np.full([self.NP, task.D], 0.0)
		return X, X_f, {'v': v}

	def runIteration(self, task: Task, X: np.ndarray, X_f: np.ndarray, xb: np.ndarray, fxb: float, v: np.ndarray, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of GravitationalSearchAlgorithm algorithm.

		Args:
			task: Optimization task.
			X: Current population.
			X_f: Current populations fitness/function values.
			xb: Global best solution.
			fxb: Global best fitness/function value.
			v: Velocity of planes.
			**dparams: Additional arguments.

		Returns:
			1. New population.
			2. New populations fitness/function values.
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments:
				* v (numpy.ndarray[float]): Velocity of planets.
		"""
		ib, iw = np.argmin(X_f), np.argmax(X_f)
		m = (X_f - X_f[iw]) / (X_f[ib] - X_f[iw])
		M = m / np.sum(m)
		Fi = np.asarray([[self.G(task.Iters) * ((M[i] * M[j]) / (self.d(X[i], X[j]) + self.epsilon)) * (X[j] - X[i]) for j in range(len(M))] for i in range(len(M))])
		F = np.sum(self.rand([self.NP, task.D]) * Fi, axis=1)
		a = F.T / (M + self.epsilon)
		v = self.rand([self.NP, task.D]) * v + a.T
		X = np.apply_along_axis(task.repair, 1, X + v, self.Rand)
		X_f = np.apply_along_axis(task.eval, 1, X)
		xb, fxb = self.getBest(X, X_f, xb, fxb)
		return X, X_f, xb, fxb, {'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
