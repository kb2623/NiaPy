# encoding=utf8

"""Implementations of Powell benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import powell_function

__all__ = ["Powell"]

class Powell(Benchmark):
	r"""Implementations of Powell functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Levy Function

		:math:`f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-4, 5]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right) \end{equation}

		Domain:
			$-4 \leq x_i \leq 5$

	Reference:
		https://www.sfu.ca/~ssurjano/levy.html
	"""
	Name: List[str] = ["Powell"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -4.0, Upper: Union[int, float, np.ndarray] = 5.0) -> None:
		r"""Initialize Powell benchmark.

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
		return r"""$f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: powell_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
