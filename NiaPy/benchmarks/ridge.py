# encoding=utf8

"""Implementation of Ridge benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import ridge_function

__all__ = ["Ridge"]

class Ridge(Benchmark):
	r"""Implementation of Ridge function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Ridge function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-64, 64]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 \end{equation}

		Domain:
			$-64 \leq x_i \leq 64$

	Reference:
		http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html
	"""
	Name: List[str] = ["Ridge"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -64.0, Upper: Union[int, float, np.ndarray] = 64.0) -> None:
		"""Initialize Ridge benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray], float]: Evaluation function.
		"""
		return lambda x, **a: ridge_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
