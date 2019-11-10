# encoding=utf8

"""Implementation of Quing benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import qing_function

__all__ = ["Qing"]

class Qing(Benchmark):
	r"""Implementation of Qing function.

	Date:
		2018

	Author:
		Lucija Brezočnik

	License:
		MIT

	Function:
		Qing function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \left(x_i^2 - i\right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (\pm √i))`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \left{(x_i^2 - i\right)}^2 \end{equation}

		Domain:
			$-500 \leq x_i \leq 500$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Qing"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -500.0, Upper: Union[int, float, np.ndarray] = 500.0) -> None:
		"""Initialize Quing benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			 Evaluation function.
		"""
		return lambda x, **a: qing_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
