# encoding=utf8
from __future__ import annotations
import logging
from typing import Dict, Union, Iterable, Tuple, List, Callable, Optional, Generator, Any

import numpy as np

from NiaPy.util.utility import objects2array
from NiaPy.util.exception import FesException, GenException, TimeException, RefException
from NiaPy.util import Task

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = ['Algorithm', 'Individual', 'defaultIndividualInit', 'defaultNumPyInit']

def defaultNumPyInit(task: Task, NP: int, rnd: np.random.RandomState = np.random.RandomState(None), **kwargs) -> Tuple[np.ndarray, np.ndarray]:
	r"""Initialize starting population that is represented with `numpy.ndarray` with shape `{NP, task.D}`.

	Args:
		task: Optimization task.
		NP: Number of individuals in population.
		rnd: Random number generator.
		kwargs: Additional arguments.

	Returns:
		1. New population with shape `{NP, task.D}`.
		2. New population function/fitness values.
	"""
	pop = task.Lower + rnd.rand(NP, task.D) * task.bRange
	fpop = np.apply_along_axis(task.eval, 1, pop)
	return pop, fpop

def defaultIndividualInit(task: Task, NP: int, rnd: np.random.RandomState = np.random.RandomState(None), itype: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
	r"""Initialize `NP` individuals of type `itype`.

	Args:
		task: Optimization task.
		NP: Number of individuals in population.
		rnd: Random number generator.
		itype: Class of individual in population.
		kwargs: Additional arguments.

	Returns:
		1. Initialized individuals.
		2. Initialized individuals function/fitness values.
	"""
	pop = objects2array([itype(task=task, rnd=rnd, e=True) for _ in range(NP)])
	return pop, np.asarray([x.f for x in pop])

class Algorithm:
	r"""Class for implementing algorithms.

	Date:
		2018

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name: List of names for algorithm.
		Rand: Random generator.
		NP: Number of inidividuals in populatin.
		InitPopFunc: Idividual initialization function.
		itype: Type of individuals used in population, default value is None for Numpy arrays.
	"""
	Name: List[str] = ['Algorithm', 'AAA']
	Rand: np.random.RandomState = np.random.RandomState(None)
	NP: int = 50
	InitPopFunc: Callable[[Task, int, np.random.RandomState, Dict[str, Any]], Tuple[np.ndarray, np.ndarray]] = defaultNumPyInit
	itype: Any = None

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get information of algorithm.

		Returns:
			str: Basic information of the algorithm.
		"""
		return r"""Base class for implementing algorithms."""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Any], bool]]:
		r"""Return functions for checking values of parameters.

		Return:
			* NP (Callable[[int], bool]): Check if number of individuals is :math:`\in [0, \infty]`.
		"""
		return {'NP': lambda x: isinstance(x, int) and x >= 1}

	def __init__(self, seed: Optional[int] = None, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		Args:
			seed: Starting seed for random generator.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		self.Rand = np.random.RandomState(seed)
		self.setParameters(**kwargs)

	def setParameters(self, NP: int = 50, InitPopFunc: Callable[[Task, int, np.random.RandomState, Dict[str, Any]], Tuple[np.ndarray, np.ndarray]] = defaultNumPyInit, itype: Optional[Any] = None, **kwargs: dict) -> None:
		r"""Set the parameters/arguments of the algorithm.

		Args:
			NP: Number of individuals in population :math:`\in [1, \infty]`.
			InitPopFunc: Type of individuals used by algorithm.
			itype: Individual type used in population, default is Numpy array.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.defaultNumPyInit`
			* :func:`NiaPy.algorithms.defaultIndividualInit`
		"""
		self.NP, self.InitPopFunc, self.itype = NP, InitPopFunc, itype
		if kwargs: logger.info('Unused arguments: %s' % (kwargs))

	def getParameters(self) -> Dict[str, Union[int, float, np.ndarray]]:
		r"""Get value of parameters for this instance of algorithm.

		Returns:
			Dictionary which has parameters mapped to values.
		"""
		return {'NP': self.NP}

	def rand(self, D: Union[np.ndarray, int, List[int], Tuple[int]] = 1) -> Union[np.ndarray, float]:
		r"""Get random distribution of shape D in range from 0 to 1.

		Args:
			D: Shape of returned random distribution.

		Returns:
			Random number or numbers :math:`\in [0, 1]`.
		"""
		if isinstance(D, (np.ndarray, list, tuple)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower: Iterable[float], Upper: Iterable[float], D: Optional[Union[int, Iterable[int]]] = None) -> Union[np.ndarray, float]:
		r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

		Args:
			Lower: Lower bound.
			Upper: Upper bound.
			D: Shape of returned uniform random distribution.

		Returns:
			Array of numbers :math:`\in [\mathit{Lower}, \mathit{Upper}]`.
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc: float, scale: float, D: Optional[Union[int, Iterable[int]]] = None) -> Union[np.ndarray, float]:
		r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

		Args:
			loc: Mean of the normal random distribution.
			scale: Standard deviation of the normal random distribution.
			D: Shape of returned normal random distribution.

		Returns:
			Array of numbers.
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D: Optional[Union[int, Iterable[int]]] = None) -> Union[np.ndarray, float]:
		r"""Get standard normal distribution of shape D.

		Args:
			D: Shape of returned standard normal distribution.

		Returns:
			Random generated numbers or one random generated number :math:`\in [0, 1]`.
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax: int, D: int = 1, Nmin: Union[int, Iterable[int]] = 0, skip: Optional[Union[int, Iterable[int], np.ndarray]] = None) -> Union[int, np.ndarray]:
		r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

		Args:
			Nmin: Lower integer bound.
			Nmax: One above upper integer bound.
			D: shape of returned discrete uniform random distribution.
			skip: numbers to skip.

		Returns:
			Random generated integer number.
		"""
		r = None
		if isinstance(D, (list, tuple, np.ndarray)): r = self.Rand.randint(Nmin, Nmax, D)
		elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
		else: r = self.Rand.randint(Nmin, Nmax)
		return r if skip is None or r not in skip else self.randint(Nmax, D, Nmin, skip)

	def getBest(self, X: np.ndarray, X_f: np.ndarray, xb: Optional[np.ndarray] = None, xb_f: float = np.inf) -> Tuple[np.ndarray, float]:
		r"""Get the best individual for population.

		Args:
			X: Current population.
			X_f: Current populations fitness/function values of aligned individuals.
			xb: Best individual.
			xb_f: Fitness value of best individual.

		Returns:
			1. Coordinates of best solution.
			2. beset fitness/function value.
		"""
		ib = np.argmin(X_f)
		if isinstance(X_f, (float, int)) and xb_f >= X_f: xb, xb_f = X.copy(), X_f
		elif isinstance(X_f, (np.ndarray, list)) and xb_f >= X_f[ib]: xb, xb_f = X[ib].copy(), X_f[ib]
		return (xb.x if isinstance(xb, Individual) else xb), xb_f

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
		r"""Initialize starting population of optimization algorithm.

		Args:
			task: Optimization task.

		Returns:
			1. New population.
			2. New population fitness values.
			3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		pop, fpop = self.InitPopFunc(task=task, NP=self.NP, rnd=self.Rand, itype=self.itype)
		return pop, fpop, {}

	def runIteration(self, task: Task, pop: np.ndarray, fpop: np.ndarray, xb: np.ndarray, fxb: float, **dparams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task: Optimization task.
			pop: Current population coordinates.
			fpop: Current population fitness value.
			xb: Current generation best individuals coordinates.
			fxb: current generation best individuals fitness value.
			**dparams: Additional arguments for algorithms.

		Returns:
			1. New populations coordinates.
			2. New populations fitness values.
			3. New global best individual.
			4. New global best individuals fitness/function value.
			5. Additional arguments of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		return pop, fpop, xb, fxb, {}

	def runYield(self, task: Task) -> Generator[Tuple[np.ndarray, float], Tuple[np.ndarray, float], None]:
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task: Task with bounds and objective function for optimization.

		Returns:
			Generator getting new/old optimal global values.

		Yield:
			1. New population best individuals coordinates.
			2. Fitness value of the best solution.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
			* :func:`NiaPy.algorithms.Algorithm.runIteration`
		"""
		pop, fpop, dparams = self.initPopulation(task)
		xb, fxb = self.getBest(pop, fpop)
		yield xb, fxb
		while True:
			pop, fpop, xb, fxb, dparams = self.runIteration(task, pop, fpop, xb, fxb, **dparams)
			yield xb, fxb

	def runTask(self, task: Task) -> Tuple[np.ndarray, float]:
		r"""Start the optimization.

		Args:
			task: Task with bounds and objective function for optimization.

		Returns:
			1. Best individuals components found in optimization process.
			2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		algo, xb, fxb = self.runYield(task), None, np.inf
		while not task.stopCond():
			xb, fxb = next(algo)
			task.nextIter()
		return xb, fxb

	def run(self, task: Task) -> Tuple[np.ndarray, float]:
		r"""Start the optimization.

		Args:
			task: Optimization task.

		Returns:
			1. Best individuals components found in optimization process.
			2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runTask`
		"""
		try:
			# task.start()
			r = self.runTask(task)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		except Exception as e: return np.full(task.D, task.Upper), np.inf * task.optType.value

class Individual:
	r"""Class that represents one solution in population of solutions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		x: Coordinates of individual.
		f: Function/fitness value of individual.
	"""
	x: Optional[np.ndarray] = None
	f: float = np.inf

	def __init__(self, x: Optional[np.ndarray] = None, task: Task = None, e: bool = True, rnd: np.random.RandomState = np.random.RandomState(None), **kwargs) -> None:
		r"""Initialize new individual.

		Parameters:
			task: Optimization task.
			rand: Random generator.
			x: Individuals components.
			e: True to evaluate the individual on initialization. Default value is True.
			**kwargs: Additional arguments.
		"""
		self.f = task.optType.value * np.inf if task is not None else np.inf
		if x is not None: self.x = x if isinstance(x, np.ndarray) else np.asarray(x)
		else: self.generateSolution(task, rnd)
		if e and task is not None: self.evaluate(task, rnd)

	def generateSolution(self, task: Task, rnd: np.random.RandomState = np.random.RandomState(None)) -> None:
		r"""Generate new solution.

		Generate new solution for this individual and set it to ``self.x``.
		This method uses ``rnd`` for getting random numbers.
		For generating random components ``rnd`` and ``task`` is used.

		Args:
			task: Optimization task.
			rnd: Random numbers generator object.
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task: Task, rnd: np.random.RandomState = np.random.RandomState) -> None:
		r"""Evaluate the solution.

		Evaluate solution ``this.x`` with the help of task.
		Task is used for reparing the solution and then evaluating it.

		Args:
			task: Objective function object.
			rnd: Random generator.

		See Also:
			* :func:`NiaPy.util.Task.repair`
		"""
		self.x = task.repair(self.x, rnd=rnd)
		self.f = task.eval(self.x)

	def attrs(self) -> Dict[str, Any]:
		r"""Get attributes of object.

		Returns:
			Attributes of object.
		"""
		return {'x': self.x.copy(), 'f': self.f}

	def copy(self) -> Individual:
		r"""Return a copy of self.

		Method returns copy of ``this`` object so it is safe for editing.

		Returns:
			Copy of self.
		"""
		return Individual(**self.attrs())

	def __eq__(self, other: Union[Individual, np.ndarray]) -> bool:
		r"""Compare the individuals for equalities.

		Args:
			other: Object that we want to compare this object to.

		Returns:
			`True` if equal or `False` if no equal.
		"""
		if isinstance(other, np.ndarray):
			for e in other:
				if self == e: return True
			return False
		return np.array_equal(self.x, other.x) and self.f == other.f

	def __str__(self) -> str:
		r"""Print the individual with the solution and objective value.

		Returns:
			String representation of self.
		"""
		return '%s -> %s' % (self.x, self.f)

	def __getitem__(self, i: int) -> Any:
		r"""Get the value of i-th component of the solution.

		Args:
			i: Position of the solution component.

		Returns:
			Value of ith component.
		"""
		return self.x[i]

	def __setitem__(self, i: int, v: Any) -> None:
		r"""Set the value of i-th component of the solution to v value.

		Args:
			i: Position of the solution component.
			v: Value to set to i-th component.
		"""
		self.x[i] = v

	def __len__(self) -> int:
		r"""Get the length of the solution or the number of components.

		Returns:
			Number of components.
		"""
		return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
