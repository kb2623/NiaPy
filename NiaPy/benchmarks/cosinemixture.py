# encoding=utf8

"""Implementation of Cosine mixture benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import cosinemixture_function

__all__ = ["CosineMixture"]

class CosineMixture(Benchmark):
	r"""Implementations of Cosine mixture function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Cosine Mixture Function

		:math:`f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		Global maximu:
			:math:`f(x^*) = -0.1 D`, at :math:`x^* = (0.0,...,0.0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$

		Equation:
			\begin{equation} f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2 \end{equation}

		Domain:
			$-1 \leq x_i \leq 14

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture
	"""
	Name: List[str] = ["CosineMixture"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -1.0, Upper: Union[int, float, np.ndarray] = 1.0) -> None:
		r"""Initialize Cosine mixture benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: cosinemixture_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
