# encoding=utf8
from typing import Callable, Tuple, Union, List, Dict
import logging

import numpy as np
from scipy.stats import levy

from NiaPy.util import Task
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CuckooSearch']

class CuckooSearch(Algorithm):
	r"""Implementation of Cuckoo behaviour and levy flights.

	Algorithm:
		Cuckoo Search

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference:
		Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009.

	Attributes:
		Name: list of strings representing algorithm names.
		N: Population size.
		pa: Proportion of worst nests.
		alpha: Scale factor for levy flight.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name: List[str] = ['CuckooSearch', 'CS']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of CuckooSearch algorithm.

		Returns:
			Basic information.
		"""
		return r"""Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009."""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[int, float]], bool]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* N
			* pa
			* alpha
		"""
		return {
			'N': lambda x: isinstance(x, int) and x > 0,
			'pa': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, (float, int)),
		}

	def setParameters(self, N: int = 50, pa: float = 0.2, alpha: float = 0.5, **ukwargs: dict) -> None:
		r"""Set the arguments of an algorithm.

		Arguments:
			N: Population size :math:`\in [1, \infty)`
			pa: factor :math:`\in [0, 1]`
			alpah: Scale factor for levy function.
			**ukwargs: Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=N, **ukwargs)
		self.pa, self.alpha = pa, alpha

	def getParameters(self) -> Dict[str, Union[int, float, np.ndarray]]:
		r"""Get parameters values.

		Returns:
			Dictionary with mapping values to parameter names.
		"""
		d = Algorithm.getParameters(self)
		d.update({'N': d.pop('NP'), 'pa': self.pa, 'alpha': self.alpha})
		return d

	def emptyNests(self, pop: np.ndarray, fpop: np.ndarray, pa_v: int, xb: np.ndarray, fxb: float, task: Task) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
		r"""Empty ensts.

		Args:
			pop: Current population
			fpop: Current population fitness/funcion values
			pa_v: Number of good nests.
			xb: Global best position.
			fxb: Global best positions function/fitness value.
			task: Optimization task

		Returns:
			1. New population
			2. New population fitness/function values
			3. New global best position.
			4. New global best positions function/fitness value.
		"""
		si = np.argsort(fpop)[:int(pa_v):-1]
		pop[si] = task.Lower + self.rand(task.D) * task.bRange
		fpop[si] = np.apply_along_axis(task.eval, 1, pop[si])
		xb, fxb = self.getBest(pop[si], fpop[si], xb, fxb)
		return pop, fpop, xb, fxb

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, dict]:
		r"""Initialize starting population.

		Args:
			task: Optimization task.

		Returns:
			1. Initialized population.
			2. Initialized populations fitness/function values.
			3. Additional arguments:
				* pa_v: Number of good nests.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		N, N_f, d = Algorithm.initPopulation(self, task)
		d.update({'pa_v': int(self.NP * self.pa)})
		return N, N_f, d

	def runIteration(self, task: Task, pop: np.ndarray, fpop: np.ndarray, xb: np.ndarray, fxb: float, pa_v: int, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of CuckooSearch algorithm.

		Args:
			task: Optimization task.
			pop: Current population.
			fpop: Current populations fitness/function values.
			xb: Global best individual.
			fxb: Global best individual function/fitness values.
			pa_v: Number of good nests.
			**dparams: Additional arguments.

		Returns:
			1. Initialized population.
			2. Initialized populations fitness/function values.
			3. Global best position.
			4. Global best position function/fitness value.
			5. Additional arguments:
				* pa_v: Number of good nests.
		"""
		i = self.randint(self.NP)
		Nn = task.repair(pop[i] + self.alpha * levy.rvs(size=[task.D], random_state=self.Rand), rnd=self.Rand)
		Nn_f = task.eval(Nn)
		j = self.randint(self.NP)
		while i == j: j = self.randint(self.NP)
		if Nn_f <= fpop[j]:
			pop[j], fpop[j] = Nn, Nn_f
			if Nn_f < fxb: xb, fxb = Nn.copy(), Nn_f
		pop, fpop, xb, fxb = self.emptyNests(pop, fpop, pa_v, xb, fxb, task)
		return pop, fpop, xb, fxb, {'pa_v': pa_v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
