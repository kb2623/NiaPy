# encoding=utf8

"""Implementation of DeflectedCorrugatedSpring benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import deflected_corrugated_spring_function

__all__ = ["DeflectedCorrugatedSpring"]

class DeflectedCorrugatedSpring(Benchmark):
	r"""Implementations of Deflected Corrugated Spring functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Deflected Corrugated Spring Function

		:math:`f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 2 \pi]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			when :math:`K = 5` and :math:`\alpha = 5` -> :math:`f(x^*) = -1`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right) \end{equation}

		Domain:
			$0 \leq x_i \leq 2 \pi$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.DeflectedCorrugatedSpring
	"""
	Name: List[str] = ["DeflectedCorrugatedSpring"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -0.0, alpha: Union[int, float, np.ndarray] = 5, K: float =5) -> None:
		r"""Initialize HGBat benchmark.

		Args:
			Lower: Lower bound of problem.
			alpha: Parameter for function and upper limit.
			K: Parameter of function.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.alpha, self.K = alpha, K
		Benchmark.__init__(self, Lower, 2 * alpha)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = 0.1 \sum_{i=1}^N \left( (x_i - \alpha)^2 - \cos \left( K \sum_{i=1}^N (x_i - \alpha)^2 \right) \right)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray], float]: Evaluation function.
		"""
		return lambda x, **a: deflected_corrugated_spring_function(x, self.alpha, self.K)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
