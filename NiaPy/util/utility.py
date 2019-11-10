# encoding=utf8
"""Implementation of benchmarks utility function."""
from typing import Union, Any, List, Tuple, Iterable

import logging
# from datetime import datetime
from enum import Enum

import numpy as np
from numpy import random as rand
from sklearn.preprocessing import LabelEncoder

from NiaPy.benchmarks import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedSchafferF6, ModifiedSchwefel, Weierstrass, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, SchafferN2, SchafferN4, LennardJones, Easom, DeflectedCorrugatedSpring, NeedleEye, Exponential, XinSheYang01, XinSheYang02, XinSheYang03, XinSheYang04, YaoLiu09, Deb01, Deb02, Bohachevsky, Tchebychev, Hilbert

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = ['Utility', 'limitRepair', 'limitInversRepair', 'wangRepair', 'randRepair', 'reflectRepair', 'fullArray', 'objects2array', 'OptimizationType', 'classifie', 'clusters2labels', 'groupdatabylabel']

def fullArray(a: Union[Tuple[Any], List[Any], Iterable[Any], np.ndarray], D: Union[int, List[int], Tuple[int], np.ndarray]) -> np.ndarray:
	r"""Fill or create array of length D, from value or value form a.

	Arguments:
		a: Input values for fill.
		D: Length of new array.

	Returns:
		Array filled with passed values or value.
	"""
	A = []
	if isinstance(a, (int, float)):	A = np.full(D, a)
	elif isinstance(a, (np.ndarray, list, tuple)):
		if len(a) == D: A = a if isinstance(a, np.ndarray) else np.asarray(a)
		elif len(a) > D: A = a[:D] if isinstance(a, np.ndarray) else np.asarray(a[:D])
		else:
			for i in range(int(np.ceil(float(D) / len(a)))): A.extend(a[:D if (D - i * len(a)) >= len(a) else D - i * len(a)])
			A = np.asarray(A)
	return A

def objects2array(objs: Union[Tuple[Any], List[Any], Iterable[Any]]) -> np.ndarray:
	r"""Convert `Iterable` array or list to `NumPy` array.

	Args:
		objs (Iterable[Any]): Array or list to convert.

	Returns:
		numpy.ndarray: Array of objects.
	"""
	a = np.empty(len(objs), dtype=object)
	for i, e in enumerate(objs): a[i] = e
	return a

class Utility:
	r"""Base class with string mappings to benchmarks.

	Attributes:
		classes (Dict[str, Benchmark]): Mapping from stings to benchmark.
	"""
	def __init__(self) -> None:
		r"""

		"""
		self.classes = {
			'ackley': Ackley,
			'alpine1': Alpine1,
			'alpine2': Alpine2,
			'bentcigar': BentCigar,
			'chungReynolds': ChungReynolds,
			'cosinemixture': CosineMixture,
			'csendes': Csendes,
			'discus': Discus,
			'dixonprice': DixonPrice,
			'conditionedellptic': Elliptic,
			'elliptic': Elliptic,
			'expandedgriewankplusrosenbrock': ExpandedGriewankPlusRosenbrock,
			'expandedschaffer': ExpandedSchafferF6,
			'griewank': Griewank,
			'happyCat': HappyCat,
			'hgbat': HGBat,
			'infinity': Infinity,
			'katsuura': Katsuura,
			'levy': Levy,
			'michalewicz': Michalewichz,
			'modifiedscwefel': ModifiedSchwefel,
			'perm': Perm,
			'pinter': Pinter,
			'powell': Powell,
			'qing': Qing,
			'quintic': Quintic,
			'rastrigin': Rastrigin,
			'ridge': Ridge,
			'rosenbrock': Rosenbrock,
			'salomon': Salomon,
			'schaffer2': SchafferN2,
			'schaffer4': SchafferN4,
			'schumerSteiglitz': SchumerSteiglitz,
			'schwefel': Schwefel,
			'schwefel221': Schwefel221,
			'schwefel222': Schwefel222,
			'sphere': Sphere,
			'sphere2': Sphere2,
			'sphere3': Sphere3,
			'step': Step,
			'step2': Step2,
			'step3': Step3,
			'stepint': Stepint,
			'styblinskiTang': StyblinskiTang,
			'trid': Trid,
			'weierstrass': Weierstrass,
			'whitley': Whitley,
			'zakharov': Zakharov,
			'lennardjones': LennardJones,
			'easom': Easom,
			'deflectedcorrugatedspring': DeflectedCorrugatedSpring,
			'needleeye': NeedleEye,
			'exponential': Exponential,
			'xinsheyang01': XinSheYang01,
			'xinsheyang02': XinSheYang02,
			'xinsheyang03': XinSheYang03,
			'xinsheyang04': XinSheYang04,
			'yaoliu09': YaoLiu09,
			'deb01': Deb01,
			'deb02': Deb02,
			'bohachevsky': Bohachevsky,
			'tchebychev': Tchebychev,
			'hilbert': Hilbert,
		}

	def get_benchmark(self, benchmark):
		r"""Get the optimization problem.

		Arguments:
			benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.

		Returns:
			Benchmark: Optimization function with limits.
		"""
		if not isinstance(benchmark, str) and not callable(benchmark): return benchmark
		elif benchmark in self.classes: return self.classes[benchmark]()
		else: raise TypeError('Passed benchmark is not defined!')

	@classmethod
	def __raiseLowerAndUpperNotDefined(cls):
		r"""Trow exception if lower and upper bounds are not defined in benchmark.

		Raises:
			TypeError: Type error.
		"""
		raise TypeError('Upper and Lower value must be defined!')

class OptimizationType(Enum):
	r"""Enum representing type of optimization.

	Attributes:
		MINIMIZATION: Represents minimization problems and is default optimization type of all algorithms.
		MAXIMIZATION: Represents maximization problems.
	"""
	MINIMIZATION: float = 1.0
	MAXIMIZATION: float = -1.0

def limitRepair(x: np.ndarray, Lower: np.ndarray, Upper: np.ndarray, **kwargs: dict) -> np.ndarray:
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x: Solution to check and repair if needed.
		Lower: Lower bounds of search space.
		Upper: Upper bounds of search space.
		kwargs: Additional arguments.

	Returns:
		Solution in search space.
	"""
	ir = np.where(x < Lower)
	x[ir] = Lower[ir]
	ir = np.where(x > Upper)
	x[ir] = Upper[ir]
	return x

def limitInversRepair(x: np.ndarray, Lower: np.ndarray, Upper: np.ndarray, **kwargs: dict) -> np.ndarray:
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x: Solution to check and repair if needed.
		Lower: Lower bounds of search space.
		Upper: Upper bounds of search space.
		kwargs: Additional arguments.

	Returns:
		Solution in search space.
	"""
	ir = np.where(x < Lower)
	x[ir] = Upper[ir]
	ir = np.where(x > Upper)
	x[ir] = Lower[ir]
	return x

def wangRepair(x: np.ndarray, Lower: np.ndarray, Upper: np.ndarray, **kwargs: dict) -> np.ndarray:
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x: Solution to check and repair if needed.
		Lower: Lower bounds of search space.
		Upper: Upper bounds of search space.
		kwargs: Additional arguments.

	Returns:
		Solution in search space.
	"""
	ir = np.where(x < Lower)
	x[ir] = np.fmin([Upper[ir], 2 * Lower[ir] - x[ir]], axis=0)
	ir = np.where(x > Upper)
	x[ir] = np.fmax([Lower[ir], 2 * Upper[ir] - x[ir]], axis=0)
	return x

def randRepair(x: np.ndarray, Lower: np.ndarray, Upper: np.ndarray, rnd: np.random.RandomState = rand, **kwargs: dict) -> np.ndarray:
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x: Solution to check and repair if needed.
		Lower: Lower bounds of search space.
		Upper: Upper bounds of search space.
		rnd: Random generator.
		kwargs: Additional arguments.

	Returns:
		Fixed solution.
	"""
	ir = np.where(x < Lower)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	ir = np.where(x > Upper)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	return x

def reflectRepair(x: np.ndarray, Lower: np.ndarray, Upper: np.ndarray, **kwargs: dict) -> np.ndarray:
	r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.

	Args:
		x: Solution to be fixed.
		Lower: Lower bounds of search space.
		Upper: Upper bounds of search space.
		kwargs: Additional arguments.

	Returns:
		Fix solution.
	"""
	ir = np.where(x > Upper)
	x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
	ir = np.where(x < Lower)
	x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
	return x

def groupdatabylabel(data: np.ndarray, labels: np.ndarray, lt: LabelEncoder) -> np.ndarray:
	r"""Get gruped data based on labels.

	Args:
		data: Dataset of individuals.
		labels: Labels of individuals.
		lt: Label transformer.

	Returns:
		Gruped data based on labels.
	"""
	G = [[] for _ in range(len(np.unique(labels)))]
	for i, e in enumerate(data): G[lt.transform([labels[i]])[0]].append(e)
	return np.asarray(G)

def clusters2labels(G_c: np.ndarray, G_l: np.ndarray) -> np.ndarray:
	r"""Get mapping from clusters to classes/labels.

	Args:
		G_c: Clusters centers.
		G_l: Centers of labeld data.

	Returns:
		Labels maped to clusters.
	"""
	a, G_ll, inds = np.full(len(G_c), -1), [gl for gl in G_l], [i for i in range(len(G_l))]
	for i, gc in enumerate(G_c):
		e = np.argmin([np.sqrt(np.sum((gc - np.mean(gl, axis=0)) ** 2)) for gl in G_ll])
		a[i] = inds[e]
		del G_ll[e]
		del inds[e]
	return a

def classifie(o: np.ndarray, C: np.ndarray) -> int:
	r"""Classfie individua based on centers.

	Args:
		o: Individual to classifie.
		C: Center of clusters.

	Returns:
		Index of class.
	"""
	return np.argmin([np.sqrt(np.sum((o - c) ** 2)) for c in C])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
