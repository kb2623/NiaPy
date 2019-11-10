# encoding=utf8

"""Implementations of Schaffer benchmarks."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import expanded_scaffer6_function, schaffern2_function, schaffern4_function

__all__ = ["SchafferN2", "SchafferN4", "ExpandedSchafferF6"]

class SchafferN2(Benchmark):
	r"""Implementations of Schaffer N. 2 functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Schaffer N. 2 Function

		:math:`f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$

		Equation:
			\begin{equation} f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 } \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""

	Name: List[str] = ["SchafferN2"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Schaffer N. 2  benchmark.

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
		return r"""$f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: schaffern2_function(x)

class SchafferN4(Benchmark):
	r"""Implementations of Schaffer N. 4 functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Schaffer N. 4 Function

		:math:`f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$

		Equation:
			\begin{equation} f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 } \end{equation}

		Domain:
			-100 \leq x_i \leq 100

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name: List[str] = ["SchafferN4"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Schaffer N. 4 benchmark.

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
		return r"""$f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: schaffern4_function(x)

class ExpandedSchafferF6(Benchmark):
	r"""Implementations of Expanded Schaffer F6 functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
		Expanded Schaffer F6 Function

		:math:`f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)^2}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
			$f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)^2}$

		Equation:
			\begin{equation} f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)^2} \end{equation}

		Domain:
			$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name: List[str] = ["ExpandedSchafferF6"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		r"""Initialize Expanded Schaffer 6 benchmark.

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
		return r"""$f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)^2}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda sol, **a: expanded_scaffer6_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
