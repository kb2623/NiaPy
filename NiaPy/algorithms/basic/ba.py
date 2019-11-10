# encoding=utf8
import logging
from typing import Tuple, Dict, List, Callable, Any

import numpy as np

from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import Task

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']

class BatAlgorithm(Algorithm):
	r"""Implementation of Bat algorithm.

	Algorithm:
		Bat algorithm

	Date:
		2015

	Authors:
		Iztok Fister Jr., Marko Burjek and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.

	Attributes:
		Name: List of strings representing algorithm name.
		A: Loudness.
		r: Pulse rate.
		Qmin: Minimum frequency.
		Qmax: Maximum frequency.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name: List[str] = ['BatAlgorithm', 'BA']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of BatAlgorithm algorithm.

		Returns:
			Basic information.
		"""
		return r"""Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74."""

	@staticmethod
	def typeParameters() -> Dict[Callable[[Any], bool], Any]:
		r"""Return dict with where key of dict represents parameter name and values represent checking functions for selected parameter.

		Returns:
			* A (Callable[[Union[float, int]], bool]): Loudness.
			* r (Callable[[Union[float, int]], bool]): Pulse rate.
			* Qmin (Callable[[Union[float, int]], bool]): Minimum frequency.
			* Qmax (Callable[[Union[float, int]], bool]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'A': lambda x: isinstance(x, (float, int)) and x > 0,
			'r': lambda x: isinstance(x, (float, int)) and x > 0,
			'Qmin': lambda x: isinstance(x, (float, int)),
			'Qmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def setParameters(self, NP: int = 40, A: float = 0.5, r: float = 0.5, Qmin: float = 0.0, Qmax: float = 2.0, **ukwargs) -> None:
		r"""Set the parameters of the algorithm.

		Args:
			A: Loudness.
			r: Pulse rate.
			Qmin: Minimum frequency.
			Qmax: Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.A, self.r, self.Qmin, self.Qmax = A, r, Qmin, Qmax

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
		r"""Initialize the starting population.

		Parameters:
			task: Optimization task

		Returns:
			1. New population.
			2. New population fitness/function values.
			3. Additional arguments:
				* S (numpy.ndarray): TODO
				* Q (numpy.ndarray[float]): 	TODO
				* v (numpy.ndarray[float]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Sol, Fitness, d = Algorithm.initPopulation(self, task)
		S, Q, v = np.full([self.NP, task.D], 0.0), np.full(self.NP, 0.0), np.full([self.NP, task.D], 0.0)
		d.update({'S': S, 'Q': Q, 'v': v})
		return Sol, Fitness, d

	def localSearch(self, best: np.ndarray, task: Task, **kwargs) -> np.ndarray:
		r"""Improve the best solution according to the Yang (2010).

		Args:
			best: Global best individual.
			task: Optimization task.
			**kwargs: Additional arguments.

		Returns:
			New solution based on global best individual.
		"""
		return task.repair(best + 0.001 * self.normal(0, 1, task.D), rnd=self.Rand)

	def runIteration(self, task: Task, Sol: np.ndarray, Fitness: np.ndarray, best: np.ndarray, f_min: float, S: np.ndarray, Q: np.ndarray, v: np.ndarray, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
		r"""Core function of Bat Algorithm.

		Parameters:
			task: Optimization task.
			Sol: Current population
			Fitness: Current population fitness/funciton values
			best: Current best individual
			f_min: Current best individual function/fitness value
			S: TODO
			Q: TODO
			v: TODO
			dparams: Additional algorithm arguments

		Returns:
			1. New population
			2. New population fitness/function vlues
			3. Additional arguments:
				* S: TODO
				* Q: TODO
				* v: TODO
		"""
		for i in range(self.NP):
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - best) * Q[i]
			S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			if self.rand() > self.r: S[i] = self.localSearch(best=best, task=task, i=i, Sol=Sol)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < self.A): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= f_min: best, f_min = S[i], Fnew
		return Sol, Fitness, best, f_min, {'S': S, 'Q': Q, 'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
