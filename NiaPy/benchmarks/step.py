# encoding=utf8

"""Implementations of Step benchmarks."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import step_function, step2_function, step3_function

__all__ = ["Step", "Step2", "Step3"]

class Step(Benchmark):
	r"""Implementation of Step function.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Step function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left | x_i \right | \rfloor \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left | x_i \right | \rfloor \right)$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left | x_i \right | \rfloor \right) \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference paper:
		 Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Step"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Step benchmark.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left | x_i \right | \rfloor \right)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: step_function(x)

class Step2(Benchmark):
	r"""Step2 function implementation.

	Date:
		2018

	Author:
		Klemen Berkovič

	Licence:
		MIT

	Function:
		Step2 function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (-0.5,...,-0.5)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2 \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Step2"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Step2 benchmark.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: step2_function(x)

class Step3(Benchmark):
	r"""Step3 function implementation.

	Date:
		2018

	Author:
		Lucija Brezočnik and Klemen Berkovic

	Licence:
		MIT

	Function:
		Step3 function

		:math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)$

		Equation:
			\begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)\end{equation}

	Domain:
		:math:`-100 \leq x_i \leq 100`

	Reference paper:
		Jamil, M., and Yang, X. S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.
	"""
	Name: List[str] = ["Step3"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Step3 benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

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
		return r"""$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: step3_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
