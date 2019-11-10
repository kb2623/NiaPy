# encoding=utf8

"""Implementations of Sphere benchmarks."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import sphere_function, sphere2_function, sphere3_function

__all__ = ["Sphere", "Sphere2", "Sphere3"]

class Sphere(Benchmark):
	r"""Implementation of Sphere functions.

	Date:
		2018

	Authors:
		Iztok Fister Jr.

	License:
		MIT

	Function:
		Sphere function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

		Domain:
			$0 \leq x_i \leq 10$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Sphere"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5.12, Upper: Union[int, float, np.ndarray] = 5.12) -> None:
		r"""Initialize Sphere benchmark.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: sphere_function(x)

class Sphere2(Benchmark):
	r"""Implementation of Sphere with different powers function.

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Function:
		Sun of different powers function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1}$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1} \end{equation}

		Domain:
			$-1 \leq x_i \leq 1$

	Reference URL:
		https://www.sfu.ca/~ssurjano/sumpow.html
	"""
	Name: List[str] = ["Sphere2"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -1.0, Upper: Union[int, float, np.ndarray] = 1.0) -> None:
		r"""Initialize Sphere benchmark.

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
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: sphere2_function(x)

class Sphere3(Benchmark):
	r"""Implementation of rotated hyper-ellipsoid function.

	Date: 2018

	Authors: Klemen Berkovič

	License: MIT

	Function:
		Sun of rotated hyper-elliposid function

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-65.536, 65.536]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$

		Equation:
			\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2 \end{equation}

		Domain:
			$-65.536 \leq x_i \leq 65.536$

	Reference URL:
		https://www.sfu.ca/~ssurjano/rothyp.html
	"""
	Name: List[str] = ["Sphere3"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -65.536, Upper: Union[int, float, np.ndarray] = 65.536) -> None:
		r"""Initialize Sphere3 benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: sphere3_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
