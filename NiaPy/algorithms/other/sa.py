# encoding=utf8
import logging

from numpy import exp

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing', 'coolDelta', 'coolLinear']

def coolDelta(currentT, T, deltaT, nFES, **kwargs):
	r"""Calculate new temperature by differences.

	Args:
		currentT (float): Current temperature.
		T (float): Max temperature.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - deltaT

def coolLinear(currentT, T, nFES, **kwargs):
	r"""Calculate temperature with linear function.

	Args:
		currentT (float): Current temperature.
		T (float): Max temperature.
		deltaT (float): Change in temperature.
		nFES (int): Number of evaluations done.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		float: New temperature.
	"""
	return currentT - T / nFES

class SimulatedAnnealing(Algorithm):
	r"""Implementation of Simulated Annealing Algorithm.

	Algorithm:
		Simulated Annealing Algorithm

	Date:
		2018

	Authors:
		Jan Popič and Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://pdfs.semanticscholar.org/e893/4a942f06ee91940ab57732953ec6a24b3f00.pdf

	Reference paper:
		S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi, “Optimization by simulated annealing,” Science, vol. 220, no. 4598, pp. 671–680, 1983.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		delta (float): Movement for neighbour search.
		T (float); Starting temperature.
		deltaT (float): Change in temperature.
		coolingMethod (Callable[[float, float, Dict[str, Any]], float]): Neighbourhood function.
		epsilon (float): Error value.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['SimulatedAnnealing', 'SA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi, “Optimization by simulated annealing,” Science, vol. 220, no. 4598, pp. 671–680, 1983."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* delta (Callable[[Union[float, int], bool])
				* T (Callable[[Union[float, int]], bool])
				* deltaT (Callable[[Union[float, int]], bool])
				* epsilon (Callable[Union[float, int]], bool])

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'delta': lambda x: isinstance(x, (int, float)) and x > 0,
			'T': lambda x: isinstance(x, (int, float)) and x > 0,
			'deltaT': lambda x: isinstance(x, (int, float)) and x > 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		})
		return d

	def setParameters(self, delta=0.5, T=2000, deltaT=0.8, coolingMethod=coolDelta, epsilon=1e-23, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:
			delta (Optional[float]): Movement for neighbour search.
			T (Optional[float]); Starting temperature.
			deltaT (Optional[float]): Change in temperature.
			coolingMethod (Optional[Callable]): Neighbourhood function.
			epsilon (Optional[float]): Error value.

		See Also
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=1, **ukwargs)
		self.delta, self.T, self.deltaT, self.cool, self.epsilon = delta, T, deltaT, coolingMethod, epsilon

	def initPopulation(self, task):
		x, xf, d = Algorithm.initPopulation(self, task)
		d.update({'curT': self.T})
		return (x[0], xf[0], d) if len(x.shape) > 1 else (x, xf, d)

	def runIteration(self, task, x, xf, xb, fxb, curT, **dparams):
		c = task.repair(x[0] - self.delta / 2 + self.rand(task.D) * self.delta, rnd=self.Rand)
		cf = task.eval(c)
		deltaFit, r = cf - xf, self.rand()
		if deltaFit < 0 or r < exp(deltaFit / curT): x, xf = c, cf
		curT = self.cool(curT, self.T, deltaT=self.deltaT, nFES=task.nFES)
		if xf <= fxb: xb, fxb = x, xf
		return x, xf, xb, fxb, {'curT': curT}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
