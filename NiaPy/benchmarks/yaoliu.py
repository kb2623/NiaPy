# encoding=utf8

"""Implementation of YaoLiu benchmark."""
from typing import List, Union, Callable

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import yaoliu_09_function

__all__ = ["YaoLiu09"]

class YaoLiu09(Benchmark):
	r"""Implementations of YaoLin09 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		YaoLiu09 Function

		:math:`f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10 \end{equation}

		Domain:
			$-5.12 \leq x_i \leq 5.12$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_Y.html#go_benchmark.YaoLiu09
	"""
	Name: List[str] = ["YaoLiu09"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5.12, Upper: Union[int, float, np.ndarray] = 5.12) -> None:
		r"""Initialize YaoLiu09 benchmark.

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
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = \sum_{i=1}^N x_i^2 - 10 \cos (2 \pi x_i) + 10$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: yaoliu_09_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
