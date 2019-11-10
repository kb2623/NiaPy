# encoding=utf8

"""Implementation of Easom benchmark."""
from typing import Callable, Union, List

import numpy as np

from math import pi

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import easom_function

__all__ = ["Easom"]

class Easom(Benchmark):
	r"""Implementations of Easom functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Easom Function

		:math:`f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			when :math:`a = 20`, :math:`b = 0.2` and :math:`c = 2 \pi` -> :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}$

		Equation:
			\begin{equation} f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)} \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_E.html#go_benchmark.Easom
	"""
	Name: List[str] = ["Easom"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0, a: float = 20, b: float = 0.2, c: float = 2 * pi) -> None:
		r"""Initialize Easom benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			a: Parameter of function.
			b: Parameter of function.
			c: Parameter of function.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.a, self.b, self.c = a, b, c
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = a - \frac{a}{e^{b \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}}} + e - e^{\frac{1}{N} \sum_{i=1}^N \cos(c x_i)}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x: easom_function(x, self.a, self.b, self.c)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
