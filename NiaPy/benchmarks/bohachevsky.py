# encoding=utf8

"""Implementation of HGBat benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import bohachevsky_function

__all__ = ["Bohachevsky"]

class Bohachevsky(Benchmark):
	r"""Implementations of Bohachevsky functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Bohachevsky Function

		:math:`f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-15, 15]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7$

		Equation:
			\begin{equation}  f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7 \end{equation}

		Domain:
			$-15 \leq x_i \leq 15$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bohachevsky
	"""
	Name: List[str] = ["Bohachevsky"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -15.0, Upper: Union[int, float, np.ndarray] = 15.0) -> None:
		r"""Initialize Bohachevsky benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			latex code.
		"""
		return r"""f(\mathbf{x}) = \sum_{i=1}^{N-1} x_i^2 + 2 x_{i+1}^2 - 0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i+1}) + 0.7"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: bohachevsky_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
