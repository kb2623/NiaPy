# encoding=utf8
from typing import Dict, Union, Callable, Tuple
import logging

import numpy as np

from NiaPy.util import objects2array, Task
from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution, CrossBest1, CrossRand1, CrossCurr2Best1, CrossBest2, CrossCurr2Rand1, proportional, multiMutations, DynNpDifferentialEvolution

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'SelfAdaptiveDifferentialEvolution',
	'DynNpSelfAdaptiveDifferentialEvolutionAlgorithm',
	'AgingSelfAdaptiveDifferentialEvolution',
	'MultiStrategySelfAdaptiveDifferentialEvolution',
	'DynNpMultiStrategySelfAdaptiveDifferentialEvolution'
]

class SolutionjDE(Individual):
	r"""Individual for jDE algorithm.

	Attributes:
		F (float): Scale factor.
		CR (float): Crossover probability.

	See Also:
		:class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, F=2, CR=0.5, **kwargs):
		r"""Initialize SolutionjDE.

		Attributes:
			F (float): Scale factor.
			CR (float): Crossover probability.

		See Also:
			:func:`NiaPy.algorithm.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.F, self.CR = F, CR

class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Self-adaptive differential evolution algorithm.

	Algorithm:
		Self-adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Uros Mlakar and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.

	Attributes:
		Name (List[str]): List of strings representing algorithm name
		F_l (float): Scaling factor lower limit.
		F_u (float): Scaling factor upper limit.
		Tao1 (float): Change rate for F parameter update.
		Tao2 (float): Change rate for CR parameter update.

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['SelfAdaptiveDifferentialEvolution', 'jDE']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of SelfAdaptiveDifferentialEvolution.

		Returns:
			Basic information.
		"""
		return r"""Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006."""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[float, int]], bool]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* F_l
			* F_u
			* Tao1
			* Tao2

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.typeParameters`
		"""
		d = DifferentialEvolution.typeParameters()
		d['F_l'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['F_u'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['Tao1'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		d['Tao2'] = lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		return d

	def setParameters(self, F_l: float = 0.0, F_u: float = 1.0, Tao1: float = 0.4, Tao2: float = 0.2, **ukwargs: dict) -> None:
		r"""Set the parameters of an algorithm.

		Arguments:
			F_l: Scaling factor lower limit.
			F_u: Scaling factor upper limit.
			Tao1: Change rate for F parameter update.
			Tao2: Change rate for CR parameter update.

		See Also:
			* :func:`NiaPy.algorithms.basic.DifferentialEvolution.setParameters`
		"""
		DifferentialEvolution.setParameters(self, itype=ukwargs.pop('itype', SolutionjDE), **ukwargs)
		self.F_l, self.F_u, self.Tao1, self.Tao2 = F_l, F_u, Tao1, Tao2

	def getParameters(self) -> Dict[str, Union[int, float, np.ndarray]]:
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dictionary which has parameters mapped to values.
		"""
		d = DifferentialEvolution.getParameters(self)
		d.pop('F', None)
		d.update({'F_l': self.F_l, 'F_u': self.F_u, 'Tao1': self.Tao1, 'Tao2': self.Tao2})
		return d

	def AdaptiveGen(self, x: SolutionjDE) -> SolutionjDE:
		r"""Adaptive update scale factor in crossover probability.

		Args:
			x: Individual to apply function on.

		Returns:
			New individual with new parameters
		"""
		f = self.F_l + self.rand() * (self.F_u - self.F_l) if self.rand() < self.Tao1 else x.F
		cr = self.rand() if self.rand() < self.Tao2 else x.CR
		return self.itype(x=x.x.copy(), F=f, CR=cr, e=False)

	def evolve(self, pop: np.ndarray, xb: np.ndarray, fxb: float, task: Task, **ukwargs: dict) -> Tuple[np.ndarray, np.ndarray, float]:
		r"""Evolve current population.

		Args:
			pop: Current population.
			xb: Global best position.
			fxb: Global best function value.
			task: Optimization task.
			ukwargs: Additional arguments.

		Returns:
			1. New population.
			2. New global best position
			3. New global best function/fitness value
		"""
		npop = objects2array([self.AdaptiveGen(e) for e in pop])
		for i, e in enumerate(npop):
			npop[i].x = self.CrossMutt(npop, i, xb, e.F, e.CR, rnd=self.Rand)
			npop[i].evaluate(task, rnd=self.rand)
			if npop[i].f <= fxb: xb, fxb = npop[i].x.copy(), npop[i].f
		return npop, xb, fxb

class AgingIndividualJDE(SolutionjDE):
	r"""Individual with age.

	Attributes:
		age (int): Age of individual.

	See Also:
		* :func:`NiaPy.algorithms.modified.SolutionjDE`
	"""
	def __init__(self, **kwargs):
		r"""Initialize aging individual for jDE algorithm.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.SolutionjDE.__init__`
		"""
		SolutionjDE.__init__(self, **kwargs)
		self.age = 0

class AgingSelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
	r"""Implementation of Dynamic population size with aging self-adaptive differential evolution algorithm.

	Algorithm:
		Dynamic population size with aging self-adaptive self adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/article/10.1007/s10489-007-0091-x

	Reference paper:
		Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
	"""
	Name = ['AgingSelfAdaptiveDifferentialEvolution', 'ANpjDE']

	@staticmethod
	def typeParameters():
		d = SelfAdaptiveDifferentialEvolution.typeParameters()
		# TODO
		return d

	def setParameters(self, LT_min=1, LT_max=7, age=proportional, **ukwargs):
		r"""Set core parameters of AgingSelfAdaptiveDifferentialEvolution algorithm.

		Args:
			LT_min (Optional[int]): Minimum age.
			LT_max (Optional[int]): Maximum age.
			age (Optional[Callable[[], int]]): Function for calculating age of individual.
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`SelfAdaptiveDifferentialEvolution.setParameters`
		"""
		SelfAdaptiveDifferentialEvolution.setParameters(self, **ukwargs)
		self.LT_min, self.LT_max, self.age = LT_min, LT_max, age
		self.mu = abs(self.LT_max - self.LT_min) / 2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

class DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(SelfAdaptiveDifferentialEvolution, DynNpDifferentialEvolution):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm.

	Algorithm:
		Dynamic population size self-adaptive differential evolution algorithm

	Date:
		2018

	Author:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://link.springer.com/article/10.1007/s10489-007-0091-x

	Reference paper:
		Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		rp (int): Small non-negative number which is added to value of generations.
		pmax (int): Number of population reductions.

	See Also:
		* :class:`NiaPy.algorithms.modified.SelfAdaptiveDifferentialEvolution`
	"""
	Name = ['DynNpSelfAdaptiveDifferentialEvolutionAlgorithm', 'dynNPjDE']

	@staticmethod
	def algorithmInfo():
		return r"""Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247."""

	@staticmethod
	def typeParameters():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		d = SelfAdaptiveDifferentialEvolution.typeParameters()
		d['rp'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['pmax'] = lambda x: isinstance(x, int) and x > 0
		return d

	def setParameters(self, rp=0, pmax=10, **ukwargs):
		r"""Set the parameters of an algorithm.

		Arguments:
			rp (Optional[int]): Small non-negative number which is added to value of genp (if it's not divisible).
			pmax (Optional[int]): Number of population reductions.

		See Also:
			* :func:`NiaPy.algorithms.modified.SelfAdaptiveDifferentialEvolution.setParameters`
		"""
		DynNpDifferentialEvolution.setParameters(self, rp=rp, pmax=pmax, **ukwargs)
		SelfAdaptiveDifferentialEvolution.setParameters(self, **ukwargs)
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def postSelection(self, pop, task, **kwargs):
		r"""Post selection operator.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		return DynNpDifferentialEvolution.postSelection(self, pop, task, **kwargs)

class MultiStrategySelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
	r"""Implementation of self-adaptive differential evolution algorithm with multiple mutation strategys.

	Algorithm:
		Self-adaptive differential evolution algorithm with multiple mutation strategys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name

	See Also:
		* :class:`NiaPy.algorithms.modified.SelfAdaptiveDifferentialEvolution`
	"""
	Name = ['MultiStrategySelfAdaptiveDifferentialEvolution', 'MsjDE']

	def setParameters(self, strategies=(CrossCurr2Rand1, CrossCurr2Best1, CrossRand1, CrossBest1, CrossBest2), **kwargs):
		r"""Set core parameters of MultiStrategySelfAdaptiveDifferentialEvolution algorithm.

		Args:
			strategys (Optional[Iterable[Callable]]): Mutations strategies to use in algorithm.
			**kwargs:

		See Also:
			* :func:`NiaPy.algorithms.modified.SelfAdaptiveDifferentialEvolution.setParameters`
		"""
		SelfAdaptiveDifferentialEvolution.setParameters(self, CrossMutt=kwargs.pop('CrossMutt', multiMutations), **kwargs)
		self.strategies = strategies

	def evolve(self, pop, xb, fxb, task, **kwargs):
		r"""Evolve population with the help multiple mutation strategies.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			xb (numpy.ndarray): Global best position.
			fxb (float): Global best function/fitness value.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New population of individuals.
		"""
		npop = []
		for i in range(len(pop)):
			nx = self.CrossMutt(pop, i, xb, self.F, self.CR, self.Rand, task, self.itype, self.strategies)
			if nx.f <= fxb: xb, fxb = nx.x, nx.f
			npop.append(nx)
		return objects2array(npop), xb, fxb

class DynNpMultiStrategySelfAdaptiveDifferentialEvolution(MultiStrategySelfAdaptiveDifferentialEvolution, DynNpSelfAdaptiveDifferentialEvolutionAlgorithm):
	r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategies.

	Algorithm:
		Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategies

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.modified.MultiStrategySelfAdaptiveDifferentialEvolution`
		* :class:`NiaPy.algorithms.modified.DynNpSelfAdaptiveDifferentialEvolutionAlgorithm`
	"""
	Name = ['DynNpMultiStrategySelfAdaptiveDifferentialEvolution', 'dynNpMsjDE']

	def setParameters(self, pmax=10, rp=5, **kwargs):
		r"""Set core parameters for algorithm instance.

		Args:
			pmax (Optional[int]):
			rp (Optional[int]):
			**kwargs (Dict[str, Any]):

		See Also:
			* :func:`NiaPy.algorithms.modified.MultiStrategySelfAdaptiveDifferentialEvolution.setParameters`
		"""
		MultiStrategySelfAdaptiveDifferentialEvolution.setParameters(self, **kwargs)
		self.pmax, self.rp = pmax, rp

	def postSelection(self, pop, task, **kwargs):
		return DynNpSelfAdaptiveDifferentialEvolutionAlgorithm.postSelection(self, pop, task)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
