# encoding=utf8

"""Implemenatation of Rastrigin benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import rastrigin_function

__all__ = ["Rastrigin"]

class Rastrigin(Benchmark):
	r"""Implementation of Rastrigin benchmark function.

	Date:
		2018

	Authors:
		Klemen Brekovič

	License:
		MIT

	Function:
		Rastrigin function

		:math:`f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right) \end{equation}

		Domain:
			$-5.12 \leq x_i \leq 5.12$

	Reference:
		https://www.sfu.ca/~ssurjano/rastr.html
	"""
	Name: List[str] = ['Rastrigin']

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5.12, Upper: Union[int, float, np.ndarray] = 5.12) -> None:
		"""Initialize Rastrigin benchmark.

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
		return r'''$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda sol, **a: rastrigin_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
