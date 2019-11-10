# encoding=utf8

"""Implementation of Chung Reynolds benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import chungreynolds_function

__all__ = ["ChungReynolds"]

class ChungReynolds(Benchmark):
	r"""Implementation of Chung Reynolds functions.

	Date:
		2018

	Authors:
		Klemen Brekovič

	License:
		MIT

	Function:
		Chung Reynolds function

		:math:`f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$

		Equation:
			\begin{equation} f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2 \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference paper:
		 Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["ChungReynolds"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		"""Initialize Chung Reynolds benchmark.

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
		return r'''$f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: chungreynolds_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
