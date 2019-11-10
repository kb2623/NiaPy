# encoding=utf8

"""Implementation of Styblinski Tang benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import styblinskitang_function

__all__ = ["StyblinskiTang"]

class StyblinskiTang(Benchmark):
	r"""Implementation of Styblinski-Tang functions.

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Function:
		Styblinski-Tang function

		:math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5, 5]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = -78.332`, at :math:`x^* = (-2.903534,...,-2.903534)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) $

		Equation:
			\begin{equation}f(\mathbf{x}) =
					\frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) \end{equation}

		Domain:
			$-5 \leq x_i \leq 5$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["StyblinskiTang"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5.0, Upper: Union[int, float, np.ndarray] = 5.0) -> None:
		r"""Initialize Styblinski Tang benchmark.

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
		return r"""$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) $"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: styblinskitang_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
