# encoding=utf8

"""Implementation of Deb benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import deb01_function, deb02_function

__all__ = ["Deb01", "Deb02"]

class Deb01(Benchmark):
	r"""Implementations of Deb01 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Deb01 Function

		:math:`f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi x_i)^6`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, with :math:`5^N` minimas

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi x_i)^6$

		Equation:
			\begin{equation} f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi x_i)^6 \end{equation}

		Domain:
			$-1 \leq x_i \leq 1$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Deb01
	"""
	Name: List[str] = ["Deb01"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -1.0, Upper: Union[int, float, np.ndarray] = 1.0) -> None:
		r"""Initialize Deb01 benchmark.

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
		return r"""f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi x_i)^6"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: deb01_function(x)

class Deb02(Benchmark):
	r"""Implementations of Deb01 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Deb02 Function

		:math:`f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi (x_i^{3/4} - 0.05))^6`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 1]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, with :math:`5^N` minimas

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi (x_i^{3/4} - 0.05))^6$

		Equation:
			\begin{equation} f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi (x_i^{3/4} - 0.05))^6\end{equation}

		Domain:
			$0 \leq x_i \leq 1$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_D.html#go_benchmark.Deb02
	"""
	Name: List[str] = ["Deb02"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = 0.0, Upper: Union[int, float, np.ndarray] = 1.0) -> None:
		r"""Initialize Deb02 benchmark.

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
		return r"""f(\mathbf{x}) = -\frac{1}{N} \sum_{i=1}^N \sin(5 \pi (x_i^{3/4} - 0.05))^6"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: deb02_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
