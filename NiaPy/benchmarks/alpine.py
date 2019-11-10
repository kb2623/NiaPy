# encoding=utf8

"""Implementations of Alpine benchmarks."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import alpine1_function, alpine2_function

__all__ = ["Alpine1", "Alpine2"]

class Alpine1(Benchmark):
	r"""Implementation of Alpine1 function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Alpine1 function

		:math:`f(\mathbf{x}) = \sum_{i=1}^{D} |x_i \sin(x_i)+0.1x_i|`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^{D} \left |x_i \sin(x_i)+0.1x_i \right|$

		Equation:
			\begin{equation} f(x) = \sum_{i=1}^{D} \left|x_i \sin(x_i) + 0.1x_i \right| \end{equation}

		Domain:
			$`-10 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Alpine1"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -10.0, Upper: Union[int, float, np.ndarray] = 10.0):
		r"""Initialize of Alpine1 benchmark.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^{D} \left |x_i \sin(x_i)+0.1x_i \right|$"""

	@classmethod
	def function(cls) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: alpine1_function(x)

class Alpine2(Benchmark):
	r"""Implementation of Alpine2 function.

	Date:
		2018

	Author:
		Klemen Brekovič

	License:
		MIT

	Function:
		Alpine2 function

		:math:`f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 2.808^D`, at :math:`x^* = (7.917,...,7.917)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i) \end{equation}

		Domain:
			$0 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Alpine2"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = 0.0, Upper: Union[int, float, np.ndarray] = 10.0) -> None:
		r"""Initialize Alpine2 benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower=Lower, Upper=Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: alpine2_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
