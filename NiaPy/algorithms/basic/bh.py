# encoding=utf8
from typing import Union, List, Dict, Callable, Tuple
import logging

import numpy as np

from NiaPy.util import Task
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BlackHole']

class BlackHole(Algorithm):
	r"""Implementation of Black Hole algorithm.

	Algorithm:
		Black hole algorithm

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		A. Hatamlou, “Black hole: a new heuristic optimization approach for data clustering,” Information Sciences, vol. 222, pp. 175–184, 2013

	Attributes:
		Name (List[str]): List of strings representing algorithm names

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name: List[str] = ['BlackHole', 'BH']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of algorithm.

		Returns:
			Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""A. Hatamlou, “Black hole: a new heuristic optimization approach for data clustering,” Information Sciences, vol. 222, pp. 175–184, 2013"""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[int, float]], bool]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* NP (Callable[[int], bool])

		See Also:
			:func:`NiaPy.algorithms.algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'vMax': lambda x: isinstance(x, (int, float))
		})
		return d

	def setParameters(self, **ukwargs: dict) -> None:
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			**ukwargs: Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, **ukwargs)

	def runIteration(self, task: Task, pop: np.ndarray, fpop: np.ndarray, xb: np.ndarray, fxb: float, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of Particle Swarm Optimization algorithm.

		Args:
			task: Optimization task.
			pop: Current populations.
			fpop: Current population fitness/function values.
			xb: Current best particle.
			fxb: Current best particle fitness/function value.
			**dparams: Additional function arguments.

		Returns:
			1. New population.
			2. New population fitness/function values.
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments:
		"""
		for i in range(len(pop)):
			pop[i] = task.repair(pop[i] + self.rand() * (xb - pop[i]), rnd=self.Rand)
			fpop[i] = task.eval(pop[i])
			if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
		r = fxb / np.sum(fpop)
		ni = np.where(r > fpop - fxb)
		if len(ni[0]) == 0: return pop, fpop, xb, fxb, {}
		for i in ni[0]: pop[i] = task.Lower + self.rand(task.D) * task.bRange
		fpop[ni] = np.apply_along_axis(task.eval, 1, pop[ni])
		ib = np.argmin(fpop[ni])
		if fpop[ni][ib] < fxb: xb, fxb = pop[ni][ib].copy(), fpop[ni][ib]
		return pop, fpop, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
