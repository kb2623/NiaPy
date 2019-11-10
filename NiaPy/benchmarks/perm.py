# encoding=utf8

"""Implementation of Perm benchmark."""
from typing import Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import perm_function

__all__ = ["Perm"]

class Perm(Benchmark):
	r"""Implementations of Perm functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Arguments:
		beta (real): Value added to inner sum of function.

	Function:
		Perm Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-D, D]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \frac{1}{2}, \cdots , \frac{1}{i} , \cdots , \frac{1}{D})`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2 \end{equation}

		Domain:
			$-D \leq x_i \leq D$

	Reference:
		https://www.sfu.ca/~ssurjano/perm0db.html
	"""
	Name: List[str] = ["Perm"]

	def __init__(self, D: float = 10.0, beta: float = 0.5):
		"""Initialize Perm benchmark.

		Args:
			D: Dimension on problem. (default: {10.0})
			beta: Beta parameter. (default: {0.5})

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.beta = beta
		Benchmark.__init__(self, -D, D)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: perm_function(x, self.beta)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
