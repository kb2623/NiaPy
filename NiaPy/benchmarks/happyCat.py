# encoding=utf8

"""Implementation of Happy Cat benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark
from .functions import happycat_function

__all__ = ["HappyCat"]

class HappyCat(Benchmark):
	r"""Implementation of Happy cat function.

	Date:
		2018

	Author:
		Lucija Brezočnik

	License:
		MIT

	Function:
		Happy cat function

		:math:`f(\mathbf{x}) = {\left |\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5`

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		Global minimum:
			:math:`f(x^*) = 0`, at :math:`x^* = (-1,...,-1)`

	LaTeX formats:
		Inline:
			$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5$

		Equation:
			\begin{equation} f(\mathbf{x}) = {\left| \sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5 \end{equation}

		Domain:
			:math:`-100 \leq x_i \leq 100`

	Reference URL:
		http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf

	Reference:
		Beyer, H. G., & Finck, S. (2012). HappyCat - A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail. In International Conference on Parallel Problem Solving from Nature (pp. 367-376). Springer, Berlin, Heidelberg.
	"""
	Name: List[str] = ["HappyCat"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -100.0, Upper: Union[int, float, np.ndarray] = 100.0) -> None:
		"""Initialize Happy Cat benchmark.

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
		return r'''$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5$'''

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda sol, **a: happycat_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
