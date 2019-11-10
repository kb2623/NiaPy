# encoding=utf8

"""Implementation of Whitley benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import whitley_function

__all__ = ["Whitley"]

class Whitley(Benchmark):
	r"""Implementation of Whitley function.

	Date:
		2018

	Authors:
		Klemne Berkovič

	License:
		MIT

	Function:
		Whitley function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10.24, 10.24]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right) \end{equation}

		Domain:
			$-10.24 \leq x_i \leq 10.24$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Whitley"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -10.24, Upper: Union[int, float, np.ndarray] = 10.24) -> None:
		"""Initialize Whitley benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: whitley_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
