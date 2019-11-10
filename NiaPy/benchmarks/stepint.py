# encoding=utf8

"""Implementation of Stepint benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import stepint_function

__all__ = ["Stepint"]


class Stepint(Benchmark):
	r"""Implementation of Stepint functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Stepint function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (-5.12,...,-5.12)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

		Domain:
			$0 \leq x_i \leq 10$

	Reference paper: Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Stepint"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5.12, Upper: Union[int, float, np.ndarray] = 5.12) -> None:
		"""Initialize Stepint benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: stepint_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
