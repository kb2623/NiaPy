# encoding=utf8

"""Implementation of Bent Cigar benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import bent_cigar_function

__all__ = ["BentCigar"]

class BentCigar(Benchmark):
	r"""Implementations of Bent Cigar functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Bent Cigar Function

		:math:`f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2$

		Equation:
			\begin{equation} f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2 \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name: List[str] = ["BentCigar"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Bent Cigar benchmark.

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
		return r"""$f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: bent_cigar_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
