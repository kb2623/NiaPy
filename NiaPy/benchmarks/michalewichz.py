# encoding=utf8

"""Implementation of Michalewichz"s benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import michalewichz_function

__all__ = ["Michalewichz"]

class Michalewichz(Benchmark):
	r"""Implementations of Michalewichz's functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Michalewichz's Function

		:math:`f(\textbf{x}) = \sum_{i=1}^D \left( 10^6 \right)^{ \frac{i - 1}{D - 1} } x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, \pi]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			* at :math:`d = 2` :math:`f(\textbf{x}^*) = -1.8013` at :math:`\textbf{x}^* = (2.20, 1.57)`
			* at :math:`d = 5` :math:`f(\textbf{x}^*) = -4.687658`
			* at :math:`d = 10` :math:`f(\textbf{x}^*) = -9.66015`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$

		Equation:
			\begin{equation} f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m} \end{equation}

		Domain:
			:math:`0 \leq x_i \leq \pi`

	Reference URL:
		https://www.sfu.ca/~ssurjano/michal.html
	"""
	Name: List[str] = ["Michalewichz"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = 0.0, Upper: Union[int, float, np.ndarray] = np.pi, m: float = 10) -> None:
		r"""Initialize Michalewichz benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			m: m attribute.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.m = m
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$"""

	def function(self) -> Callable[[np.ndarray, dict], str]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: michalewichz_function(x, self.m)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
