# encoding=utf8

"""Implementation of Xin She Yang benchmark."""
from typing import Union, Callable, List, Optional

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import xin_she_yang_01_function, xin_she_yang_02_function, xin_she_yang_03_function, xin_she_yang_04_function

__all__ = ["XinSheYang01", "XinSheYang02", "XinSheYang03", "XinSheYang04"]

class XinSheYang01(Benchmark):
	r"""Implementations of Xin She Yang 01 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Xin She Yang 01 Function

		:math:`f(\mathbf{x}) = \sum_{i=1}^N \epsilon_i | x_i |^i`

		Variables:
			* :math:`\epsilon_i`: is a randomvariable uniformly distributed and :math:`i = (1, 2 \cdots, N)` where :math:`\epsilon_i \in [0, 1]`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-5, 5]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \sum_{i=1}^N \epsilon_i | x_i |^i$

		Equation:
			\begin{equation} f(\mathbf{x}) = \sum_{i=1}^N \epsilon_i | x_i |^i \end{equation}

		Domain:
			$-5 \leq x_i \leq 5$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_X.html#go_benchmark.XinSheYang01
	"""
	Name: List[str] = ["XinSheYang01"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -5, Upper: Union[int, float, np.ndarray] = 5, epsilon: Optional[float] = None) -> None:
		r"""Initialize XinSheYang02 benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			epsilon: Some value.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		self.epsilon = epsilon
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = \sum_{i=1}^N \epsilon_i | x_i |^i$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: xin_she_yang_01_function(x, self.epsilon)

class XinSheYang02(Benchmark):
	r"""Implementations of Xin She Yang 02 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Xin She Yang 02 Function

		:math:`f(\mathbf{x}) = \frac{\sum_{i=1}^N | x_i |}{e^{\sum_{i=1}^N \sin(x_i^2)}}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-2 \pi, 2 \pi]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \frac{\sum_{i=1}^N | x_i |}{e^{\sum_{i=1}^N \sin(x_i^2)}}$

		Equation:
			\begin{equation} f(\mathbf{x}) = \frac{\sum_{i=1}^N | x_i |}{e^{\sum_{i=1}^N \sin(x_i^2)}} \end{equation}

		Domain:
			$-2 \pi \leq x_i \leq 2 \pi$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_X.html#go_benchmark.XinSheYang02
	"""
	Name: List[str] = ["XinSheYang02"]

	def __init__(self, Lower: Union[float, np.ndarray] = -2 * np.pi, Upper: Union[float, np.ndarray] = 2 * np.pi) -> None:
		r"""Initialize XinSheYang02 benchmark.

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
		return r"""$f(\mathbf{x}) = \frac{\sum_{i=1}^N | x_i |}{e^{\sum_{i=1}^N \sin(x_i^2)}}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: xin_she_yang_02_function(x)

class XinSheYang03(Benchmark):
	r"""Implementations of Xin She Yang 03 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Xin She Yang 03 Function

		:math:`f(\mathbf{x}) = e^{\sum_{i=1}^N (x_i / \beta)^{2m}} - 2 e^{-\sum_{i=1}^N x_i^2} \prod_{i=1}^N \cos(x_i)^2`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-20, 20]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = e^{\sum_{i=1}^N (x_i / \beta)^{2m}} - 2 e^{-\sum_{i=1}^N x_i^2} \prod_{i=1}^N \cos(x_i)^2$

		Equation:
			\begin{equation} f(\mathbf{x}) = e^{\sum_{i=1}^N (x_i / \beta)^{2m}} - 2 e^{-\sum_{i=1}^N x_i^2} \prod_{i=1}^N \cos(x_i)^2 \end{equation}

		Domain:
			$-20 \leq x_i \leq 20$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_X.html#go_benchmark.XinSheYang03
	"""
	Name: List[str] = ["XinSheYang03"]
	beta: float = 15
	m: float = 3

	def __init__(self, Lower: Union[int, float, np.ndarray] = -20.0, Upper: Union[int, float, np.ndarray] = 20.0, beta: float = 15, m: float = 3) -> None:
		r"""Initialize XinSheYang03 benchmark.

		Args:
			Lower: Lower bound of problem.
			Upper: Upper bound of problem.
			beta: Parameter of function.
			m: Parameter of function.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		XinSheYang03.beta, XinSheYang03.m = beta, m
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{x}) = e^{\sum_{i=1}^N (x_i / \beta)^{2m}} - 2 e^{-\sum_{i=1}^N x_i^2} \prod_{i=1}^N \cos(x_i)^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: xin_she_yang_03_function(x, self.beta, self.m)

class XinSheYang04(Benchmark):
	r"""Implementations of Xin She Yang 03 functions.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Xin She Yang 04 Function

		:math:`f(\mathbf{x}) = \left( \sum_{i=1}^N \sin(x_i)^2 - e^{-\sum_{i=1}^N x_i^2} \right) e^{-\sum_{i=1}^N \sin \left(\sqrt{| x_i |} \right)^2}`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., N`.

		Global minimum:
			:math:`f(x^*) = -1`, at :math:`x^* = (1,...,1)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = \left( \sum_{i=1}^N \sin(x_i)^2 - e^{-\sum_{i=1}^N x_i^2} \right) e^{-\sum_{i=1}^N \sin(\sqrt{| x_i |})^2}$

		Equation:
			\begin{equation} f(\mathbf{x}) = \left( \sum_{i=1}^N \sin(x_i)^2 - e^{-\sum_{i=1}^N x_i^2} \right) e^{-\sum_{i=1}^N \sin \left(\sqrt{| x_i |} \right)^2} \end{equation}

		Domain:
			$-10 \leq x_i \leq 10$

	Reference:
		http://infinity77.net/global_optimization/test_functions_nd_X.html#go_benchmark.XinSheYang04
	"""
	Name: List[str] = ["XinSheYang04"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -10.0, Upper: Union[int, float, np.ndarray] = 10.0) -> None:
		r"""Initialize XinSheYang04 benchmark.

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
		return r"""$f(\mathbf{x}) = \left( \sum_{i=1}^N \sin(x_i)^2 - e^{-\sum_{i=1}^N x_i^2} \right) e^{-\sum_{i=1}^N \sin \left(\sqrt{| x_i |} \right)^2}$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda x, **a: xin_she_yang_04_function(x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
