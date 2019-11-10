# encoding=utf8

"""Implementation of base benchmark class."""
import logging
from typing import Union, Callable, List

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

logging.basicConfig()
logger = logging.getLogger("NiaPy.benchmarks.benchmark")
logger.setLevel("INFO")

__all__ = ["Benchmark"]

class Benchmark:
	r"""Class representing benchmarks.

	Attributes:
		Name: List of names representiong benchmark names.
		Lower (Union[int, float, list, numpy.ndarray]): Lower bounds.
		Upper (Union[int, float, list, numpy.ndarray]): Upper bounds.
	"""
	Name: List[str] = ["Benchmark", "BBB"]

	def __init__(self, Lower: Union[int, float, np.ndarray], Upper: Union[int, float, np.ndarray], **kwargs) -> None:
		r"""Initialize benchmark.

		Args:
			Lower: Lower bounds.
			Upper: Upper bounds.
			**kwargs: Additional arguments.
		"""
		self.Lower = Lower
		self.Upper = Upper

	@staticmethod
	def latex_code() -> str:
		r"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""None"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		r"""Get evaluation function.

		Returns:
			Evaluation function.
		"""
		def evaluate(sol: np.ndarray, **kwargs) -> float:
			r"""Utility/Evaluation function.

			Args:
				sol: Solution to evaluate.

			Returns:
				Function value.
			"""
			return np.inf
		return evaluate

	def plot2d(self):
		"""Plot."""
		pass

	def __2dfun(self, x: float, y: float, f: Callable[[np.ndarray, dict], float]) -> float:
		r"""Calculate function value.

		Args:
			x: First coordinate.
			y: Second coordinate.
			f: Evaluation function.

		Returns:
			Calculate functional value for given input
		"""
		return f(2, x, y)

	def plot3d(self, scale: float = 0.32) -> None:
		r"""Plot 3d scatter plot of benchmark function.

		Args:
			scale: Scale factor for points.
		"""
		fig = plt.figure()
		ax = fig.gca(projection="3d")
		func = self.function()
		Xr, Yr = np.arange(self.Lower, self.Upper, scale), np.arange(self.Lower, self.Upper, scale)
		X, Y = np.meshgrid(Xr, Yr)
		Z = np.vectorize(self.__2dfun)(X, Y, func)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		ax.contourf(X, Y, Z, zdir="z", offset=-10, cmap=cm.coolwarm)
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
