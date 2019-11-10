# encoding=utf8
from __future__ import annotations
from typing import Tuple, Union, Any, Optional, Dict, Callable
import logging

import numpy as np
from numpy import random as rand

from NiaPy.algorithms.algorithm import Algorithm, Individual
from NiaPy.util.utility import objects2array

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CamelAlgorithm']

class Camel(Individual):
	r"""Implementation of population individual that is a camel for Camel algorithm.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		E: Camel endurance.
		S: Camel supply.
		x_past: Camel's past position.
		f_past: Camel's past funciton/fitness value.
		steps: Age of camel.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, E_init: Optional[float] = None, S_init: Optional[float] = None, x_past: Optional[np.ndarray] = None, f_past: Optional[float] = None, **kwargs: dict) -> None:
		r"""Initialize the Camel.

		Args:
			E_init: Starting endurance of Camel.
			S_init: Stating supply of Camel.
			x_past: Camel's past position.
			f_past: Camel's past funciton/fitness value.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.E, self.E_past = E_init, E_init
		self.S, self.S_past = S_init, S_init
		self.x_past, self.f_past = self.x if x_past is None else x_past, self.f if f_past is None else f_past
		self.steps = 0

	def nextT(self, T_min: float, T_max: float, rnd: np.random.RandomState = rand) -> None:
		r"""Apply nextT function on Camel.

		Args:
			T_min: Minimum temperature.
			T_max: Maximum temperature.
			rnd: Random number generator.
		"""
		self.T = (T_max - T_min) * rnd.rand() + T_min

	def nextS(self, omega: float, n_gens: int) -> None:
		r"""Apply nextS on Camel.

		Args:
			omega: Dying rate.
			n_gens: Number of Camel Algorithm iterations/generations.
		"""
		self.S = self.S_past * (1 - omega * self.steps / n_gens)

	def nextE(self, n_gens: int, T_max: float) -> None:
		r"""Apply function nextE on function on Camel.

		Args:
			n_gens: Number of Camel Algorithm iterations/generations
			T_max: Maximum temperature of environment
		"""
		self.E = self.E_past * (1 - self.T / T_max) * (1 - self.steps / n_gens)

	def nextX(self, cb: np.ndarray, E_init: float, S_init: float, task: Task, rnd: np.random.RandomState = rand) -> None:
		r"""Apply function nextX on Camel.

		This method/function move this Camel to new position in search space.

		Args:
			cb: Best Camel in population.
			E_init: Starting endurance of camel.
			S_init: Starting supply of camel.
			task: Optimization task.
			rnd: Random number generator.
		"""
		delta = -1 + rnd.rand() * 2
		self.x = self.x_past + delta * (1 - (self.E / E_init)) * np.exp(1 - self.S / S_init) * (cb - self.x_past)
		if not task.isFeasible(self.x): self.x = self.x_past
		else: self.f = task.eval(self.x)

	def next(self) -> None:
		r"""Save new position of Camel to old position."""
		self.x_past, self.f_past, self.E_past, self.S_past = self.x, self.f, self.E, self.S
		self.steps += 1

	def refill(self, S: Optional[float] = None, E: Optional[float] = None) -> None:
		r"""Apply this function to Camel.

		Args:
			S: New value of Camel supply.
			E: New value of Camel endurance.
		"""
		self.S, self.E = S, E

	def attrs(self) -> Dict[str, Any]:
		r"""Get attributes values of object.

		Returns:
			Dict[str, Any]: Attributes of object.
		"""
		d = Individual.attrs(self)
		d.update({'E_past': self.E_past, 'S_past': self.S_past})
		return d

	def copy(self) -> Camel:
		r"""Get a copy of camel.

		Returns:
			Camel: Copy of camel.
		"""
		return Camel(**self.attrs())

class CamelAlgorithm(Algorithm):
	r"""Implementation of Camel traveling behavior.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.iasj.net/iasj?func=fulltext&aId=118375

	Reference paper:
		Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.

	Attributes:
		Name (List[str]): List of strings representing name of the algorithm.
		T_min (float): Minimal temperature of environment.
		T_max (float): Maximal temperature of environment.
		E_init (float): Starting value of energy.
		S_init (float): Starting value of supplys.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name: List[str] = ['CamelAlgorithm', 'CA']

	@staticmethod
	def algorithmInfo() -> str:
		r"""Get basic information of CamelAlgorithm algorithm.

		Returns:
			Basic information.
		"""
		return r"""Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177."""

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[float, int]], bool]]:
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			* omega (Callable[[Union[int, float]], bool])
			* mu (Callable[[float], bool])
			* alpha (Callable[[float], bool])
			* S_init (Callable[[Union[float, int]], bool])
			* E_init (Callable[[Union[float, int]], bool])
			* T_min (Callable[[Union[float, int], bool])
			* T_max (Callable[[Union[float, int], bool])

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'omega': lambda x: isinstance(x, (float, int)),
			'mu': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'S_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'E_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_min': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_max': lambda x: isinstance(x, (float, int)) and x > 0
		})
		return d

	def setParameters(self, NP: int = 50, omega: float = 0.25, mu: float = 0.5, alpha: float = 0.5, S_init: float = 10, E_init: float = 10, T_min: float = -10, T_max: float = 10, **ukwargs: dict) -> None:
		r"""Set the arguments of an algorithm.

		Arguments:
			NP: Population size :math:`\in [1, \infty)`.
			T_min: Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
			T_max: Maximum temperature, must be true :math:`T_{min} < T_{max}`.
			omega: Burden factor :math:`\in [0, 1]`.
			mu: Dying rate :math:`\in [0, 1]`.
			S_init: Initial supply :math:`\in (0, \infty)`.
			E_init: Initial endurance :math:`\in (0, \infty)`.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, itype=Camel, InitPopFunc=ukwargs.pop('InitPopFunc', self.initPop), **ukwargs)
		self.omega, self.mu, self.alpha, self.S_init, self.E_init, self.T_min, self.T_max = omega, mu, alpha, S_init, E_init, T_min, T_max

	def initPop(self, task: Task, NP: int, rnd: np.random.RandomState, itype: Individual, **kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
		r"""Initialize starting population.

		Args:
			task: Optimization task.
			NP: Number of camels in population.
			rnd: Random number generator.
			itype: Individual type.
			**kwargs: Additional arguments.

		Returns:
			1. Initialize population of camels.
			2. Initialized populations function/fitness values.
		"""
		caravan = objects2array([itype(E_init=self.E_init, S_init=self.S_init, task=task, rnd=rnd, e=True) for _ in range(NP)])
		return caravan, np.asarray([c.f for c in caravan])

	def walk(self, c: Camel, xb: np.ndarray, fxb: float, task: Task) -> Camel:
		r"""Move the camel in search space.

		Args:
			c: Camel that we want to move.
			xb: Global best best position.
			fxb: Global best positions function/fitness value.
			task: Optimization task.

		Returns:
			Camel that moved in the search space.
		"""
		c.nextT(self.T_min, self.T_max, self.Rand)
		c.nextS(self.omega, task.nGEN)
		c.nextE(task.nGEN, self.T_max)
		c.nextX(xb, self.E_init, self.S_init, task, self.Rand)
		if c.f < fxb: xb, fxb = c.x.copy(), c.f
		return c, xb, fxb

	def oasis(self, c: Camel, rn: float, alpha: float) -> Camel:
		r"""Apply oasis function to camel.

		Args:
			c: Camel to apply oasis on.
			rn: Random number.
			alpha: View range of Camel.

		Returns:
			Camel with appliyed oasis on.
		"""
		if rn > 1 - alpha and c.f < c.f_past: c.refill(self.S_init, self.E_init)
		return c

	def lifeCycle(self, c: Camel, mu: float, task: Task) -> Camel:
		r"""Apply life cycle to Camel.

		Args:
			c: Camel to apply life cycle.
			mu: Vision range of camel.
			task: Optimization task.

		Returns:
			Camel with life cycle applyed to it.
		"""
		if c.f_past < mu * c.f: return Camel(self.E_init, self.S_init, rnd=self.Rand, task=task)
		c.next()
		return c

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, dict]:
		r"""Initialize population.

		Args:
			task: Optimization taks.

		Returns:
			1. New population of Camels.
			2. New population fitness/function values.
			3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		caravan, fcaravan, _ = Algorithm.initPopulation(self, task)
		return caravan, fcaravan, {}

	def runIteration(self, task: Task, caravan: np.ndarray, fcaravan: np.ndarray, xb: np.ndarray, fxb: float, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of Camel Algorithm.

		Args:
			task: Optimization task.
			caravan: Current population of Camels.
			fcaravan: Current population fitness/function values.
			xb: Global best position.
			fxb: Global best positions function/fitness value.
			**dparams: Additional arguments.

		Returns:
			1. New population
			2. New population function/fitness value
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments
		"""
		ncaravan = []
		for c in caravan: nc, xb, fxb = self.walk(c, xb, fxb, task); ncaravan.append(nc)
		ncaravan = objects2array(ncaravan)
		ncaravan = objects2array([self.oasis(c, self.rand(), self.alpha) for c in ncaravan])
		ncaravan = objects2array([self.lifeCycle(c, self.mu, task) for c in ncaravan])
		return ncaravan, np.asarray([x.f for x in ncaravan]), xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
