# encoding=utf8

"""Implementation of Tchebyshev benchmark."""
from typing import Union, Callable, List

import numpy as np

from NiaPy.benchmarks import Benchmark
from .functions import Tchebyshev_function

__all__ = ["Tchebychev"]

class Tchebychev(Benchmark):
	r"""Implementations of Storn's Tchebychev function.

	Storn's Tchebychev - a 2nd ICEO function - generalized version
	* Valid for any D>2
	* constraints: unconstrained
	* type: multi-modal with one global minimum; non-separable
	* initial upper bound = 2^D, initial lower bound = -D^n
	* value-to-reach = f(x*)+1.0e-8
	* f(x*)=0.0; x*=(128,0,-256,0,160,0,-32,0,1) (n=9)
	* x*=(32768,0,-131072,0,212992,0,-180224,0,84480,0,-21504,0,2688,0,-128,0,1) (n=17)

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Lennard Jones potential

		:math:``

		Input domain:
			The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\infty, \infty]`, for all :math:`i = 1, 2,..., D`.

	LaTeX formats:
		Inline:
			$$

		Equation:
			\begin{equation} \end{equation}

		Domain:
			$$

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`
	"""
	Name: List[str] = ["LennardJones"]

	def __init__(self, Lower: Union[int, float, np.ndarray] = -np.inf, Upper: Union[int, float, np.ndarray] = np.inf):
		r"""Create Tchebychev benchmark.

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
		return lambda sol: Tchebyshev_function(sol)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
