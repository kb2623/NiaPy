# encoding=utf8

"""Implementations of Weierstrass functions."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import weierstrass_function

__all__ = ["Weierstrass"]


class Weierstrass(Benchmark):
	r"""Implementations of Weierstrass functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Weierstass Function

		:math:`f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`. Default value of a = 0.5, b = 3 and k_max = 20.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$$f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right) \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		 http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""

	Name: List[str] = ["Weierstrass"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0, a: float = 0.5, b: int = 3, k_max: int = 20) -> None:
		"""Initialize Weierstrass benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			a: A value.
			b: B value
			k_max: Value.
		"""
		self.a, self.b, self.k_max = a, b, k_max
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$$f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda sol, **a: weierstrass_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
