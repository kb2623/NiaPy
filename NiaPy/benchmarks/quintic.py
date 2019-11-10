# encoding=utf8

"""Implementation of Quintic benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import quintic_function

__all__ = ["Quintic"]

class Quintic(Benchmark):
	r"""Implementation of Quintic function.

	Date:
		2018

	Author:
		Lucija Brezočnik

	License:
		MIT

	Function:
		Quintic function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4\right|`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = f(-1\; \text{or}\; 2)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4\right|$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4\right| \end{equation}

		Domain:
			$-10 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Quintic"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -10.0, Upper: Union[int, float, np.ndarray] = 10.0) -> None:
		"""Initialize Quintic benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 - 10x_i - 4\right|$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: quintic_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
