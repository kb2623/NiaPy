# encoding=utf8

"""Implementation of NeedleEye benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import needle_eye_function

__all__ = ["NeedleEye"]

class NeedleEye(Benchmark):
	r"""Implementations of NeedleEye functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Needle Eye Function

		:math:`f(\mathbf{x}) = \begin{cases} 1 & \mathrm{if} \quad |x_i| < eye \\ \sum_{i=1}^N 100 + |x_i| & \mathrm{if} \quad |x_i| > eye \\ 1 & \mathrm{otherwise} \end{cases}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			when :math:`a = 20`, :math:`b = 0.2` and :math:`c = 2 \pi` -> :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \begin{cases} 1 & \mathrm{if} \quad |x_i| < eye \\ \sum_{i=1}^N 100 + |x_i| & \mathrm{if} \quad |x_i| > eye \\ 1 & \mathrm{otherwise} \end{cases}$

		Equation:
			\begin{equation} f(\mathbf{x}) = \begin{cases} 1 & \mathrm{if} \quad |x_i| < eye \\ \sum_{i=1}^N 100 + |x_i| & \mathrm{if} \quad |x_i| > eye \\ 1 & \mathrm{otherwise} \end{cases} \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_N.html#go_benchmark.NeedleEye
	"""
	Name: List[str] = ['NeedleEye']
	eye: float = 1

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0, eye: float = 1) -> None:
		r"""Initialize HGBat benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			eye: Parameter of function.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.eye = eye
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = \begin{cases} 1 & \mathrm{if} \quad |x_i| < eye \\ \sum_{i=1}^N 100 + |x_i| & \mathrm{if} \quad |x_i| > eye \\ 1 & \mathrm{otherwise} \end{cases}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: needle_eye_function(x, self.eye)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
