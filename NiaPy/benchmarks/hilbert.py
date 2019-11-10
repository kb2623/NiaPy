# encoding=utf8

"""Implementation of Hilbert benchmark."""
from typing import Callable, Union, List

import numpy as np

from NiaPy.benchmarks import Benchmark

from .functions import Hilbert_function

__all__ = ["Hilbert"]

class Hilbert(Benchmark):
	r"""Implementations of Hilbert function.

	Find the inverse of the (ill-conditioned) Hilbert matrix
	* valid for any dimension, n=k*k, k=2,3,4,...
	* constraints: unconstrained
	* type: multi-modal with one global minimum; non-separable
	* initial upper bound = 2^n, initial lower bound = -2^n
	* value-to-reach = f(x*)+1.0e-8
	* f(x*) = 0.0; x*={{9,-36,30},{-36,192,-180},{30,-180,180}} (n=9)
	* x*={{16,-120,240,-140},{-120,1200,-2700,1680},{240,-2700,6480,4200},{-140,1680,-4200,2800}} (n=16)

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Hilbert function

		:math:``

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\infty, \infty]`, for all :math:`i = 1, 2,..., D`.

		LaTeX formats:
			Inline:
				$$

			Equation:
				\begin{equation} \end{equation}

			Domain:
				:math:``

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`
	"""
	Name: List[str] = ["Hilbert"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -np.inf, Upper: Union[int, float, np.ndarray] = np.inf) -> None:
		r"""Create Hilbert benchmark.

		Args:
			Lower: Lower bound limits.
			Upper: Upper bound limits.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper)

	@staticmethod
	def latex_code() -> str:
		r"""Get latex code for function.

		Returns:
			Latex code.
		"""
		return r"""TODO"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		r"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		return lambda sol, **a: Hilbert_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
