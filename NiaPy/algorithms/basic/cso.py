# encoding=utf8
from typing import List, Union, Callable, Dict, Tuple
import logging

import numpy as np
import math

from NiaPy.util import Task, limitRepair
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CatSwarmOptimization']

class CatSwarmOptimization(Algorithm):
	r"""Implementation of Cat swarm optimiization algorithm.

	Algorithm:
		Cat swarm optimization

	Date:
		2019

	Author:
		Mihael Baketarić and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Chu, Shu-Chuan & Tsai, Pei-Wei & Pan, Jeng-Shyang. (2006). Cat Swarm Optimization. 854-858. 10.1007/11801603_94.
	"""
	Name: List[str] = ['CatSwarmOptimization', 'CSO']

	@staticmethod
	def typeParameters() -> Dict[str, Callable[[Union[bool, int, float]], bool]]:
		return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'MR': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
			'C1': lambda x: isinstance(x, (int, float)) and x >= 0,
			'SMP': lambda x: isinstance(x, int) and x > 0,
			'SPC': lambda x: isinstance(x, bool),
			'CDC': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
			'SRD': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
			'vMax': lambda x: isinstance(x, (int, float)) and x > 0
		}

	def setParameters(self, NP: int = 30, MR: float = 0.1, C1: float = 2.05, SMP: int = 3, SPC: bool = True, CDC: float = 0.85, SRD: float = 0.2, vMax: Union[float, np.ndarray] = 1.9, **ukwargs: dict) -> None:
		r"""Set the algorithm parameters.

		Arguments:
			NP: Number of individuals in population.
			MR: Mixture ratio.
			C1: Constant in tracing mode.
			SMP: Seeking memory pool.
			SPC: Self-position considering.
			CDC: Decides how many dimensions will be varied.
			SRD: Seeking range of the selected dimension.
			vMax: Maximal velocity.
			ukwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.MR, self.C1, self.SMP, self.SPC, self.CDC, self.SRD, self.vMax = MR, C1, SMP, SPC, CDC, SRD, vMax

	def getParameters(self) -> Dict[str, Union[int, float, np.ndarray]]:
		r"""Get parameters vales.

		Returns:
			Dictionary mapping parameter name to parameter value.
		"""
		d = Algorithm.getParameters(self)
		d.update({'MR': self.MR, 'C1': self.C1, 'SMP': self.SMP, 'SPC': self.SPC, 'CDC': self.CDC, 'SRD': self.SRD, 'vMax': self.vMax})
		return d

	def initPopulation(self, task: Task) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		r"""Initialize population.

		Args:
			 task: Optimization task.

		Returns:
			1. Initialized population.
			2. Initialized populations fitness/function values.
			3. Additional arguments:
				* Dictionary of modes (seek or trace) and velocities for each cat

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		pop, fpop, d = Algorithm.initPopulation(self, task)
		d['modes'] = self.randomSeekTrace()
		d['velocities'] = self.uniform(-self.vMax, self.vMax, [len(pop), task.D])
		return pop, fpop, d

	def randomSeekTrace(self) -> np.ndarray:
		r"""Set cats into seeking/tracing mode.

		Returns:
			numpy.ndarray: One or zero. One means tracing mode. Zero means seeking mode. Length of list is equal to NP.
		"""
		lista = np.zeros((self.NP,), dtype=int)
		indexes = np.arange(self.NP)
		self.Rand.shuffle(indexes)
		lista[indexes[:int(self.NP * self.MR)]] = 1
		return lista

	def weightedSelection(self, weights: np.ndarray) -> int:
		r"""Random selection considering the weights.

		Args:
			weights: weight for each potential position.

		Returns:
			index of selected next position.
		"""
		cumulative_sum = np.cumsum(weights)
		return np.argmax(cumulative_sum >= (self.rand() * cumulative_sum[-1]))

	def seekingMode(self, task: Task, cat: np.ndarray, fcat: float, pop: np.ndarray, fpop: np.ndarray, xb: np.ndarray, fxb: float) -> Tuple[np.ndarray, float, np.ndarray, float]:
		r"""Seeking mode.

		Args:
			task: Optimization task.
			cat: Individual from population.
			fcat: Current individual's fitness/function value.
			pop: Current population.
			fpop: Current population fitness/function values.
			xb: Current best cat position.
			fxb: Current best cat fitness/function value.

		Returns:
			1. Updated individual's position
			2. Updated individual's fitness/function value
			3. Updated global best position
			4. Updated global best fitness/function value
		"""
		cat_copies = []
		cat_copies_fs = []
		for j in range(self.SMP - 1 if self.SPC else self.SMP):
			cat_copies.append(cat.copy())
			indexes = np.arange(task.D)
			self.Rand.shuffle(indexes)
			to_vary_indexes = indexes[:int(task.D * self.CDC)]
			if self.randint(2) == 1:
				cat_copies[j][to_vary_indexes] += cat_copies[j][to_vary_indexes] * self.SRD
			else:
				cat_copies[j][to_vary_indexes] -= cat_copies[j][to_vary_indexes] * self.SRD
			cat_copies[j] = task.repair(cat_copies[j])
			cat_copies_fs.append(task.eval(cat_copies[j]))
		if self.SPC:
			cat_copies.append(cat.copy())
			cat_copies_fs.append(fcat)
		cat_copies_select_probs = np.ones(len(cat_copies))
		imin = np.argmin(cat_copies_fs)
		fmax = np.max(cat_copies_fs)
		fmin = cat_copies_fs[imin]
		if any(x != cat_copies_fs[0] for x in cat_copies_fs):
			fb = fmax
			if math.isinf(fb): cat_copies_select_probs = np.full(len(cat_copies), fb)
			else: cat_copies_select_probs = np.abs(cat_copies_fs - fb) / (fmax - fmin)
		if fmin < fxb: xb, fxb = cat_copies[imin], fmin
		sel_index = self.weightedSelection(cat_copies_select_probs)
		return cat_copies[sel_index], cat_copies_fs[sel_index], xb, fxb

	def tracingMode(self, task: Task, cat: np.ndarray, velocity: np.ndarray, xb: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
		r"""Tracing mode.

		Args:
			task: Optimization task.
			cat: Individual from population.
			velocity: Velocity of individual.
			xb: Current best individual.

		Returns:
			1. Updated individual's position
			2. Updated individual's fitness/function value
			3. Updated individual's velocity vector
		"""
		Vnew = limitRepair(velocity + (self.uniform(0, 1, len(velocity)) * self.C1 * (xb - cat)), np.full(task.D, -self.vMax), np.full(task.D, self.vMax))
		cat_new = task.repair(cat + Vnew)
		return cat_new, task.eval(cat_new), Vnew

	def runIteration(self, task: Task, pop: np.ndarray, fpop: np.ndarray, xb: np.ndarray, fxb: float, velocities: np.ndarray, modes: np.ndarray, **dparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
		r"""Core function of Cat Swarm Optimization algorithm.

		Args:
			task: Optimization task.
			pop: Current population.
			fpop: Current population fitness/function values.
			xb: Current best individual.
			fxb: Current best cat fitness/function value.
			velocities: Velocities of individuals.
			modes: Flag of each individual.
			**dparams: Additional function arguments.

		Returns:
			1. New population
			2. New population fitness/function values
			3. Additional arguments:
				* Dictionary of modes (seek or trace) and velocities for each cat
		"""
		pop_copies = pop.copy()
		for k in range(len(pop_copies)):
			if modes[k] == 0:
				pop_copies[k], fpop[k], xb, fxb = self.seekingMode(task, pop_copies[k], fpop[k], pop_copies, fpop, xb, fxb)
			else:  # if cat in tracing mode
				pop_copies[k], fpop[k], velocities[k] = self.tracingMode(task, pop_copies[k], velocities[k], xb)
				if fpop[k] < fxb: xb, fxb = pop_copies[k], fpop[k]
		return pop_copies, fpop, xb, fxb, {'velocities': velocities, 'modes': self.randomSeekTrace()}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
