# encoding=utf8
from typing import Union, Tuple, List, Any, Optional, Callable
import logging

import numpy as np
from numpy import random as rand
from matplotlib import pyplot as plt, animation as anim

from NiaPy.util.utility import fullArray, Utility, OptimizationType, limitRepair
from NiaPy.util.exception import FesException, GenException, RefException  # TimeException,
from NiaPy.benchmarks.benchmark import Benchmark

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = [
	'Task',
	'CountingTask',
	'StoppingTask',
	'ThrowingTask',
	'TaskConvPrint',
	'TaskConvPlot',
	'TaskConvSave',
	'TaskComposition',
	'ScaledTask',
]

class Task(Utility):
	r"""Class representing problem to solve with optimization.

	Date:
		2019

	Author:
		Klemen Berkovič

	Attributes:
		D (int): Dimension of the problem.
		Lower (numpy.ndarray): Lower bounds of the problem.
		Upper (numpy.ndarray): Upper bounds of the problem.
		bRange (numpy.ndarray): Search range between upper and lower limits.
		benchmark (Benchmark): Used benchmark.
		optType (OptimizationType): Optimization type to use.

	See Also:
		* :class:`NiaPy.util.Utility`
	"""
	D: int = 0
	benchmark: Benchmark = None
	Lower: Union[int, float, np.ndarray] = np.inf
	Upper: Union[int, float, np.ndarray] = np.inf
	bRange: Union[int, float, np.ndarray] = np.inf
	optType: OptimizationType = OptimizationType.MINIMIZATION

	def __init__(self, D: int = 0, optType: OptimizationType = OptimizationType.MINIMIZATION, benchmark: Optional[Union[str, Benchmark]] = None, Lower: Optional[Union[int, float, np.ndarray]] = None, Upper: Optional[Union[int, float, np.ndarray]] = None, frepair: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray] = limitRepair, **kwargs: dict) -> None:
		r"""Initialize task class for optimization.

		Arguments:
			D: Number of dimensions.
			optType: Set the type of optimization.
			benchmark: Problem to solve with optimization.
			Lower: Lower limits of the problem.
			Upper: Upper limits of the problem.
			frepair: Function for reparing individuals components to desired limits.

		See Also:
			* `func`:NiaPy.util.Utility.__init__`
			* `func`:NiaPy.util.Utility.repair`
		"""
		Utility.__init__(self)
		# dimension of the problem
		self.D = D
		# set optimization type
		self.optType = optType
		# set optimization function
		self.benchmark = self.get_benchmark(benchmark) if benchmark is not None else None
		if self.benchmark is not None: self.Fun = self.benchmark.function() if self.benchmark is not None else None
		# set Lower limits
		if Lower is not None: self.Lower = fullArray(Lower, self.D)
		elif Lower is None and benchmark is not None: self.Lower = fullArray(self.benchmark.Lower, self.D)
		else: self.Lower = fullArray(0, self.D)
		# set Upper limits
		if Upper is not None: self.Upper = fullArray(Upper, self.D)
		elif Upper is None and benchmark is not None: self.Upper = fullArray(self.benchmark.Upper, self.D)
		else: self.Upper = fullArray(0, self.D)
		# set range
		self.bRange = self.Upper - self.Lower
		# set repair function
		self.frepair = frepair

	def names(self) -> List[str]:
		r"""Get list of strings representing benchmark names.

		Returns:
			Names of benchmark.
		"""
		return self.benchmark.Name

	def dim(self) -> int:
		r"""Get the number of dimensions.

		Returns:
			Dimension of problem optimizing.
		"""
		return self.D

	def bcLower(self) -> np.ndarray:
		r"""Get the array of lower bound constraint.

		Returns:
			Lower bound.
		"""
		return self.Lower

	def bcUpper(self) -> np.ndarray:
		r"""Get the array of upper bound constraint.

		Returns:
			Upper bound.
		"""
		return self.Upper

	def bcRange(self) -> np.ndarray:
		r"""Get the range of bound constraint.

		Returns:
			Range between lower and upper bound.
		"""
		return self.Upper - self.Lower

	def repair(self, x: np.ndarray, rnd: np.random.RandomState = rand) -> np.ndarray:
		r"""Repair solution and put the solution in the random position inside of the bounds of problem.

		Arguments:
			x: Solution to check and repair if needed.
			rnd: Random number generator.

		Returns:
			Fixed solution.

		See Also:
			* :func:`NiaPy.util.limitRepair`
			* :func:`NiaPy.util.limitInversRepair`
			* :func:`NiaPy.util.wangRepair`
			* :func:`NiaPy.util.randRepair`
			* :func:`NiaPy.util.reflectRepair`
		"""
		return self.frepair(x, self.Lower, self.Upper, rnd=rnd)

	def nextIter(self):
		r"""Increments the number of algorithm iterations."""

	def start(self):
		r"""Start stopwatch."""

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate the solution A.

		Arguments:
			A: Solution to evaluate.

		Returns:
			Fitness/function values of solution.
		"""
		return self.Fun(A) * self.optType.value

	def isFeasible(self, A: np.ndarray) -> bool:
		r"""Check if the solution is feasible.

		Arguments:
			A: Solution to check for feasibility.

		Returns:
			bool: `True` if solution is in feasible space else `False`.
		"""
		return False not in (A >= self.Lower) and False not in (A <= self.Upper)

	def stopCond(self) -> bool:
		r"""Check if optimization task should stop.

		Returns:
			`True` if stopping condition is meet else `False`.
		"""
		return False

class CountingTask(Task):
	r"""Optimization task with added counting of function evaluations and algorithm iterations/generations.

	Attributes:
		Iters: Number of algorithm iterations/generations.
		Evals: Number of function evaluations.

	See Also:
		* :class:`NiaPy.util.Task`
	"""
	Iters: int = 0
	Evals: int = 0

	def __init__(self, **kwargs: dict) -> None:
		r"""Initialize counting task.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.Task.__init__`
		"""
		Task.__init__(self, **kwargs)
		self.Iters, self.Evals = 0, 0

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate the solution A.

		This function increments function evaluation counter `self.Evals`.

		Arguments:
			A: Solutions to evaluate.

		Returns:
			Fitness/function values of solution.

		See Also:
			* :func:`NiaPy.util.Task.eval`
		"""
		r = Task.eval(self, A)
		self.Evals += 1
		return r

	def evals(self) -> int:
		r"""Get the number of evaluations made.

		Returns:
			Number of evaluations made.
		"""
		return self.Evals

	def iters(self) -> int:
		r"""Get the number of algorithm iteratins made.

		Returns:
			int: Number of generations/iterations made by algorithm.
		"""
		return self.Iters

	def nextIter(self) -> None:
		r"""Increases the number of algorithm iterations made.

		This function increments number of algorithm iterations/generations counter `self.Iters`.
		"""
		self.Iters += 1

class StoppingTask(CountingTask):
	r"""Optimization task with implemented checking for stopping criterias.

	Attributes:
		nGEN (int): Maximum number of algorithm iterations/generations.
		nFES (int): Maximum number of function evaluations.
		refValue (float): Reference function/fitness values to reach in optimization.
		x (numpy.ndarray): Best found individual.
		x_f (float): Best found individual function/fitness value.

	See Also:
		* :class:`NiaPy.util.CountingTask`
	"""

	def __init__(self, nFES: int = np.inf, nGEN: int = np.inf, refValue: Optional[float] = None, **kwargs: dict) -> None:
		r"""Initialize task class for optimization.

		Arguments:
			nFES: Number of function evaluations.
			nGEN: Number of generations or iterations.
			refValue: Reference value of function/fitness function.

		See Also:
			* :func:`NiaPy.util.CountingTask.__init__`
		"""
		CountingTask.__init__(self, **kwargs)
		self.refValue = (-np.inf if refValue is None else refValue)
		self.x_f = np.inf * self.optType.value
		self.nFES, self.nGEN = nFES, nGEN

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Solution to evaluate.

		Returns:
			float: Fitness/function value of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.eval`
		"""
		if self.stopCond(): return np.inf * self.optType.value
		x_f = CountingTask.eval(self, A)
		if x_f < self.x_f: self.x_f = x_f
		return x_f

	def stopCond(self) -> bool:
		r"""Check if stopping condition reached.

		Returns:
			`True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`
		"""
		return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN) or (self.refValue > self.x_f)

	def stopCondI(self) -> bool:
		r"""Check if stopping condition reached and increase number of iterations.

		Returns:
			`True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.nextIter`
		"""
		r = self.stopCond()
		CountingTask.nextIter(self)
		return r

class ThrowingTask(StoppingTask):
	r"""Task that throw exceptions when stopping condition is meet.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def __init__(self, **kwargs: dict) -> None:
		r"""Initialize optimization task.

		Args:
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.x = None

	def stopCondE(self) -> None:
		r"""Throw exception for the given stopping condition.

		Raises:
			* FesException: Thrown when the number of function/fitness evaluations is reached.
			* GenException: Thrown when the number of algorithms generations/iterations is reached.
			* RefException: Thrown when the reference values is reached.
			* TimeException: Thrown when algorithm exceeds time run limit.
		"""
		# dtime = datetime.now() - self.startTime
		if self.Evals >= self.nFES: raise FesException()
		if self.Iters >= self.nGEN: raise GenException()
		# if self.runTime is not None and self.runTime >= dtime: raise TimeException()
		if self.refValue >= self.x_f: raise RefException()

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Solution to evaluate.

		Returns:
			Function/fitness values of solution.

		See Also:
			* :func:`NiaPy.util.ThrowingTask.stopCondE`
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		self.stopCondE()
		x_f = self.x_f
		r = StoppingTask.eval(self, A)
		if r != x_f: self.x = A
		return r

class MoveTask(StoppingTask):
	def __init__(self, o: Optional[np.ndarray] = None, fo: Optional[Callable[[np.ndarray], np.ndarray]] = None, M: Optional[np.ndarray] = None, fM: Optional[Callable[[np.ndarray], np.ndarray]] = None, optF=None, **kwargs) -> None:
		r"""Initialize task class for optimization.

		Arguments:
			o: Array for shifting.
			fo: Function applied on shifted input.
			M: Matrix for rotating.
			fM: Function applied after rotating.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.o = o if isinstance(o, np.ndarray) or o is None else np.asarray(o)
		self.M = M if isinstance(M, np.ndarray) or M is None else np.asarray(M)
		self.fo, self.fM, self.optF = fo, fM, optF

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate the solution.

		Args:
			A: Solution to evaluate

		Returns:
			Fitness/function value of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		if self.stopCond(): return np.inf * self.optType.value
		X = A - self.o if self.o is not None else A
		X = self.fo(X) if self.fo is not None else X
		X = np.dot(X, self.M) if self.M is not None else X
		X = self.fM(X) if self.fM is not None else X
		r = StoppingTask.eval(self, X) + (self.optF if self.optF is not None else 0)
		if r <= self.x_f: self.x, self.x_f = A, r
		return r

class ScaledTask(Task):
	r"""Scaled task.

	Attributes:
		_task (Task): Optimization task with evaluation function.
		Lower (numpy.ndarray): Scaled lower limit of search space.
		Upper (numpy.ndarray): Scaled upper limit of search space.

	See Also:
		* :class:`NiaPy.util.Task`
	"""
	def __init__(self, task: Task, Lower: Union[int, float, np.ndarray], Upper: Union[int, float, np.ndarray], **kwargs: dict) -> None:
		r"""Initialize scaled task.

		Args:
			task: Optimization task to scale to new bounds.
			Lower: New lower bounds.
			Upper: New upper bounds.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.util.fullArray`
		"""
		Task.__init__(self)
		self._task = task
		self.D = self._task.D
		self.Lower, self.Upper = fullArray(Lower, self.D), fullArray(Upper, self.D)
		self.bRange = np.fabs(Upper - Lower)

	def stopCond(self) -> bool:
		r"""Test for stopping condition.

		This function uses `self._task` for checking the stopping criteria.

		Returns:
			`True` if stopping condition is meet else `False`.
		"""
		return self._task.stopCond()

	def stopCondI(self) -> bool:
		r"""Test for stopping condition and increments the number of algorithm generations/iterations.

		This function uses `self._task` for checking the stopping criteria.

		Returns:
			`True` if stopping condition is meet else `False`.
		"""
		return self._task.stopCondI()

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Solution for calculating function/fitness value.

		Returns:
			Function values of solution.
		"""
		return self._task.eval(A)

	def evals(self) -> int:
		r"""Get the number of function evaluations.

		Returns:
			Number of function evaluations.
		"""
		return self._task.evals()

	def iters(self) -> int:
		r"""Get the number of algorithms generations/iterations.

		Returns:
			Number of algorithms generations/iterations.
		"""
		return self._task.iters()

	def nextIter(self) -> None:
		r"""Increment the number of iterations/generations.

		Function uses `self._task` to increment number of generations/iterations.
		"""
		self._task.nextIter()

class TaskConvPrint(StoppingTask):
	r"""Task class with printing out new global best solutions found.

	Attributes:
		xb (numpy.ndarray): Global best solution.
		xb_f (float): Global best function/fitness values.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	xb: np.ndarray = None
	xb_f: float = np.inf

	def __init__(self, **kwargs: dict) -> None:
		r"""Initialize TaskConvPrint class.

		Args:
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.xb, self.xb_f = None, np.inf

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Solution to evaluate.

		Returns:
			Function/Fitness values of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		x_f = StoppingTask.eval(self, A)
		if self.x_f != self.xb_f:
			self.xb, self.xb_f = A, x_f
			logger.info('nFES:%d nGEN:%d => %s -> %s' % (self.Evals, self.Iters, self.xb, self.xb_f * self.optType.value))
		return x_f

class TaskConvSave(StoppingTask):
	r"""Task class with logging of function evaluations need to reach some function vale.

	Attributes:
		evals (List[int]): List of ints representing when the new global best was found.
		x_f_vals (List[float]): List of floats representing function/fitness values found.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	evals: List[int] = []
	x_f_vals: List[float] = []

	def __init__(self, **kwargs: dict) -> None:
		r"""Initialize TaskConvSave class.

		Args:
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.evals = []
		self.x_f_vals = []

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Individual/solution to evaluate.

		Returns:
			Function/fitness values of individual.

		See Also:
			* :func:`SNiaPy.util.toppingTask.eval`
		"""
		x_f = StoppingTask.eval(self, A)
		if x_f <= self.x_f:
			self.evals.append(self.Evals)
			self.x_f_vals.append(x_f)
		return x_f

	def return_conv(self) -> Tuple[List[int], List[float]]:
		r"""Get values of x and y axis for plotting covariance graph.

		Returns:
			1. List of ints of function evaluations.
			2. List of ints of function/fitness values.
		"""
		return self.evals, self.x_f_vals

class TaskConvPlot(TaskConvSave):
	r"""Task class with ability of showing convergence graph.

	Attributes:
		iters (List[int]): List of ints representing when the new global best was found.
		x_fs (List[float]): List of floats representing function/fitness values found.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	iters: List[int] = []
	x_fs: List[float] = []

	def __init__(self, **kwargs: dict) -> None:
		r"""TODO.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.fig = plt.figure()
		self.ax = self.fig.subplots(nrows=1, ncols=1)
		self.ax.set_xlim(0, self.nFES)
		self.line, = self.ax.plot(self.iters, self.x_fs, animated=True)
		self.ani = anim.FuncAnimation(self.fig, self.updatePlot, blit=True)
		self.showPlot()

	def eval(self, A: np.ndarray) -> float:
		r"""Evaluate solution.

		Args:
			A: Solution to evaluate.

		Returns:
			Fitness/function values of solution.
		"""
		x_f = StoppingTask.eval(self, A)
		if not self.x_f_vals: self.x_f_vals.append(x_f)
		elif x_f < self.x_f_vals[-1]: self.x_f_vals.append(x_f)
		else: self.x_f_vals.append(self.x_f_vals[-1])
		self.evals.append(self.Evals)
		return x_f

	def showPlot(self) -> None:
		r"""Animation updating function."""
		plt.show(block=False)
		plt.pause(0.001)

	def updatePlot(self, frame):
		r"""Update mathplotlib figure.

		Args:
			frame (): TODO

		Returns:
			Tuple[List[float], Any]:
				1. Line
		"""
		if self.x_f_vals:
			max_fs, min_fs = self.x_f_vals[0], self.x_f_vals[-1]
			self.ax.set_ylim(min_fs + 1, max_fs + 1)
			self.line.set_data(self.evals, self.x_f_vals)
		return self.line,

class TaskComposition(MoveTask):
	def __init__(self, benchmarks=None, rho=None, lamb=None, bias=None, **kwargs):
		r"""Initialize of composite function problem.

		Arguments:
			benchmarks (List[Benchmark]): Optimization function to use in composition
			delta (numpy.ndarray[float]): TODO
			lamb (numpy.ndarray[float]): TODO
			bias (numpy.ndarray[float]): TODO

		See Also:
			* :func:`NiaPy.util.MoveTask.__init__`

		TODO:
			Class is a work in progress.
		"""
		MoveTask.__init__(self, **kwargs)

	def eval(self, A):
		r"""TODO.

		Args:
			A:

		Returns:
			float:

		Todo:
			Usage of multiple functions on the same time
		"""
		return np.inf

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
