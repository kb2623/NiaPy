# encoding=utf8
import logging
from math import ceil

from numpy import argmin, argsort, log, sum, fmax, fmin, sqrt, full, exp, eye, diag, apply_along_axis, asarray, inf, where, append, arange, triu, isnan
from numpy.linalg import norm, eig


from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit
from NiaPy.util.utility import objects2array

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy1p1', 'EvolutionStrategyMp1', 'EvolutionStrategyMpL', 'EvolutionStrategyML', 'CovarianceMatrixAdaptionEvolutionStrategy']

class IndividualES(Individual):
	r"""Individual for Evolution Strategies.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kwargs):
		r"""Initialize individual.

		Args:
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		task, x, rho = kwargs.get('task', None), kwargs.get('x', None), kwargs.get('rho', 1)
		if rho != None: self.rho = rho
		elif task != None or x != None: self.rho = 1.0

class EvolutionStrategy1p1(Algorithm):
	r"""Implementation of (1 + 1) evolution strategy algorithm. Uses just one individual.

	Algorithm:
		(1 + 1) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		mu (int): Number of parents.
		k (int): Number of iterations before checking and fixing rho.
		c_a (float): Search range amplification factor.
		c_r (float): Search range reduction factor.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['EvolutionStrategy1p1', 'EvolutionStrategy(1+1)', 'ES(1+1)']

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* mu (Callable[[int], bool])
				* k (Callable[[int], bool])
				* c_a (Callable[[Union[float, int]], bool])
				* c_r (Callable[[Union[float, int]], bool])
				* epsilon (Callable[[float], bool])
		"""
		return {
			'mu': lambda x: isinstance(x, int) and x > 0,
			'k': lambda x: isinstance(x, int) and x > 0,
			'c_a': lambda x: isinstance(x, (float, int)) and x > 1,
			'c_r': lambda x: isinstance(x, (float, int)) and 0 < x < 1,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		}

	def setParameters(self, mu=1, k=10, c_a=1.1, c_r=0.5, epsilon=1e-20, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			mu (Optional[int]): Number of parents
			k (Optional[int]): Number of iterations before checking and fixing rho
			c_a (Optional[float]): Search range amplification factor
			c_r (Optional[float]): Search range reduction factor

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=mu, itype=ukwargs.pop('itype', IndividualES), **ukwargs)
		self.mu, self.k, self.c_a, self.c_r, self.epsilon = mu, k, c_a, c_r, epsilon

	def mutate(self, x, rho):
		r"""Mutate individual.

		Args:
			x (Individual): Current individual.
			rho (float): Current standard deviation.

		Returns:
			Individual: Mutated individual.
		"""
		return x + self.normal(0, rho, len(x))

	def updateRho(self, rho, k):
		r"""Update standard deviation.

		Args:
			rho (float): Current standard deviation.
			k (int): Number of succesfull mutations.

		Returns:
			float: New standard deviation.
		"""
		phi = k / self.k
		if phi < 0.2: return self.c_r * rho if rho > self.epsilon else 1
		elif phi > 0.2: return self.c_a * rho if rho > self.epsilon else 1
		else: return rho

	def initPopulation(self, task):
		r"""Initialize starting individual.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[Individual, float, Dict[str, Any]]:
				1, Initialized individual.
				2, Initialized individual fitness/function value.
				3. Additional arguments:
					* ki (int): Number of successful rho update.
		"""
		c, ki = IndividualES(task=task, rnd=self.Rand), 0
		return c, c.f, {'ki': ki}

	def runIteration(self, task, c, fpop, xb, fxb, ki, **dparams):
		r"""Core function of EvolutionStrategy(1+1) algorithm.

		Args:
			task (Task): Optimization task.
			pop (Individual): Current position.
			fpop (float): Current position function/fitness value.
			xb (Individual): Global best position.
			fxb (float): Global best function/fitness value.
			ki (int): Number of successful updates before rho update.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[Individual, float, numpy.ndarray, float, Dict[str, Any]]:
				1, Initialized individual.
				2, Initialized individual fitness/function value.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments:
					* ki (int): Number of successful rho update.
		"""
		if task.Iters % self.k == 0: c.rho, ki = self.updateRho(c.rho, ki), 0
		cn = objects2array([task.repair(self.mutate(c.x, c.rho), self.Rand) for _i in range(self.mu)])
		cn_f = asarray([task.eval(cn[i]) for i in range(len(cn))])
		ib = argmin(cn_f)
		if cn_f[ib] < c.f: c.x, c.f, ki = cn[ib], cn_f[ib], ki + 1
		return c, c.f, c.x, c.f, {'ki': ki}

class EvolutionStrategyMp1(EvolutionStrategy1p1):
	r"""Implementation of (mu + 1) evolution strategy algorithm. Algorithm creates mu mutants but into new generation goes only one individual.

	Algorithm:
		(:math:`\mu + 1`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
	"""
	Name = ['EvolutionStrategyMp1', 'EvolutionStrategy(mu+1)', 'ES(m+1)']

	def setParameters(self, **kwargs):
		r"""Set core parameters of EvolutionStrategy(mu+1) algorithm.

		Args:
			**kwargs (Dict[str, Any]):

		See Also:
			* :func:`NiaPy.algorithms.basic.EvolutionStrategy1p1.setParameters`
		"""
		mu = kwargs.pop('mu', 40)
		EvolutionStrategy1p1.setParameters(self, mu=mu, **kwargs)

class EvolutionStrategyMpL(EvolutionStrategy1p1):
	r"""Implementation of (mu + lambda) evolution strategy algorithm. Mulation creates lambda individual. Lambda individual compete with mu individuals for survival, so only mu individual go to new generation.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		lam (int): Number of new individuals generated by mutation.

	See Also:
		* :class:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
	"""
	Name = ['EvolutionStrategyMpL', 'EvolutionStrategy(mu+lambda)', 'ES(m+l)']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Any]:
				* lam (Callable[[int], bool]): TODO.

		See Also:
			* :func:`NiaPy.algorithms.basic.EvolutionStrategy1p1`
		"""
		d = EvolutionStrategy1p1.typeParameters()
		d['lam'] = lambda x: isinstance(x, int) and x > 0
		return d

	def setParameters(self, lam=45, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			lam (int): Number of new individual generated by mutation.

		See Also:
			* :func:`NiaPy.algorithms.basic.es.EvolutionStrategy1p1.setParameters`
		"""
		EvolutionStrategy1p1.setParameters(self, InitPopFunc=defaultIndividualInit, **ukwargs)
		self.lam = lam

	def updateRho(self, pop, k):
		r"""Update standard deviation for population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			k (int): Number of successful mutations.
		"""
		phi = k / self.k
		if phi < 0.2:
			for i in pop: i.rho = self.c_r * i.rho
		elif phi > 0.2:
			for i in pop: i.rho = self.c_a * i.rho

	def changeCount(self, c, cn):
		r"""Update number of successful mutations for population.

		Args:
			c (numpy.ndarray[Individual]): Current population.
			cn (numpy.ndarray[Individual]): New population.

		Returns:
			int: Number of successful mutations.
		"""
		k = 0
		for e in cn:
			if False in e == c: k += 1
		return k

	def mutateRand(self, pop, task):
		r"""Mutate random individual form population.

		Args:
			pop (numpy.ndarray[Individual]): Current population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Random individual from population that was mutated.
		"""
		i = self.randint(self.mu)
		return task.repair(self.mutate(pop[i].x, pop[i].rho), rnd=self.Rand)

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized populaiton.
				2. Initialized populations function/fitness values.
				3. Additional arguments:
					* ki (int): Number of successful mutations.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		c, fc, d = Algorithm.initPopulation(self, task)
		d.update({'ki': 0})
		return c, fc, d

	def runIteration(self, task, c, fpop, xb, fxb, ki, **dparams):
		r"""Core function of EvolutionStrategyMpL algorithm.

		Args:
			task (Task): Optimization task.
			c (numpy.ndarray[Individual]): Current population.
			fpop (numpy.ndarray[float]): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals fitness/function value.
			ki (int): Number of successful mutations.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations function/fitness values.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments:
					* ki (int): Number of successful mutations.
		"""
		if task.Iters % self.k == 0: _, ki = self.updateRho(c, ki), 0
		cn = objects2array([IndividualES(x=self.mutateRand(c, task), task=task, rnd=self.Rand) for _ in range(self.lam)])
		cn = append(cn, c)
		cn = objects2array([cn[i] for i in argsort([i.f for i in cn])[:self.mu]])
		fcn = asarray([x.f for x in cn])
		ki += self.changeCount(c, cn)
		xb, fxb = self.getBest(cn, fcn, xb, fxb)
		return cn, [x.f for x in cn], xb, fxb, {'ki': ki}

class EvolutionStrategyML(EvolutionStrategyMpL):
	r"""Implementation of (mu, lambda) evolution strategy algorithm. Algorithm is good for dynamic environments. Mu individual create lambda chields. Only best mu chields go to new generation. Mu parents are discarded.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:

	Reference paper:

	Attributes:
		Name (List[str]): List of strings representing algorithm names

	See Also:
		* :class:`NiaPy.algorithm.basic.es.EvolutionStrategyMpL`
	"""
	Name = ['EvolutionStrategyML', 'EvolutionStrategy(mu,lambda)', 'ES(m,l)']

	def newPop(self, pop):
		r"""Return new population.

		Args:
			pop (numpy.ndarray): Current population.

		Returns:
			numpy.ndarray: New population.
		"""
		pop_s = argsort([i.f for i in pop])
		if self.mu < self.lam: return objects2array([pop[i] for i in pop_s[:self.mu]])
		npop = list()
		for i in range(int(ceil(float(self.mu) / self.lam))): npop.extend(pop[:self.lam if (self.mu - i * self.lam) >= self.lam else self.mu - i * self.lam])
		return objects2array(npop)

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				2. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.es.EvolutionStrategyMpL.initPopulation`
		"""
		c, fc, _ = EvolutionStrategyMpL.initPopulation(self, task)
		return c, fc, {}

	def runIteration(self, task, c, fpop, xb, fxb, **dparams):
		r"""Core function of EvolutionStrategyML algorithm.

		Args:
			task (Task): Optimization task.
			c (numpy.ndarray[Individual]): Current population.
			fpop (numpy.ndarray[float]): Current population fitness/function values.
			xb (Individual): Global best individual.
			fxb (float): Global best individuals fitness/function value.
			**dparams Dict[str, Any]: Additional arguments.

		Returns:
			Tuple[numpy.ndarray[Individual], numpy.ndarray[float], numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best position.
				4. New global best positions function/fitness value.
				5. Additional arguments.
		"""
		cn = objects2array([IndividualES(x=self.mutateRand(c, task), task=task, rand=self.Rand) for _ in range(self.lam)])
		c = self.newPop(cn)
		fc = asarray([x.f for x in c])
		xb, fxb = self.getBest(c, fc, xb, fxb)
		return c, asarray([x.f for x in c]), xb, fxb, {}

class CovarianceMatrixAdaptionEvolutionStrategy(Algorithm):
	r"""Implementation of (mu, lambda) evolution strategy algorithm. Algorithm is good for dynamic environments. Mu individual create lambda chields. Only best mu chields go to new generation. Mu parents are discarded.

	Algorithm:
		(:math:`\mu + \lambda`) Evolution Strategy Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		* https://link.springer.com/chapter/10.1007/3-540-32494-1_4
		* https://arxiv.org/abs/1604.00772
		* https://en.wikipedia.org/wiki/CMA-ES
		* https://github.com/AdeelMufti/WorldModels/blob/master/toy/cma-es.py

	Reference paper:
		* Hansen, N. (2006), "The CMA evolution strategy: a comparing review", Towards a new evolutionary computation. Advances on estimation of distribution algorithms, Springer, pp. 1769–1776
		* Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).

	Attributes:
		Name (List[str]): List of names representing algorithm names
		mu (int): Number of parents.
		lambdA (int): Number of offsprings.
		sigma0 (float): Initial standard deviation.
		epsilon (float): Small number for testing if value has to be fixed.
	"""
	Name = ['CovarianceMatrixAdaptionEvolutionStrategy', 'CMA-ES', 'CMAES']
	epsilon = 1e-20

	@staticmethod
	def typeParameters(): return {
			'epsilon': lambda x: isinstance(x, (float, int)) and 0 < x < 1
	}

	def setParameters(self, mu=None, lambdA=None, sigma0=0.3, epsilon=1e-20, **ukwargs):
		r"""Set core parameters of CovarianceMatrixAdaptionEvolutionStrategy algorithm.

		Args:
			mu (Optional[int]): Number of parents.
			lambdA (Optional[int]): Number of offsprings.
			sigma0 (Optional[float]): Initial standard deviation.
			epsilon (Optional[float]): Small number for testing if value has to be fixed.
			**ukwargs (Dict[str, Any]): Additional arguments.
		"""
		Algorithm.setParameters(self, NP=1, **ukwargs)
		self.mu, self.lambdA, self.sigma0, self.epsilon = mu, lambdA, sigma0, epsilon

	def initPopulation(self, task):
		r"""Init starting population and parameters of algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, Dict[str, Any]]:
				1. Starting individual.
				2. Starting individuals function/fitness value.
				3. Additional arguments:
					* xmean (flaot): Mean over dimensions.
					* mu (int): Number of parents.
					* lambdA (int): Number of offsprings.
					* sigma (float): Coordinate wise standard deviation (step size).
					* weights (numpy.ndarray[float]): Array for weighted recombination.
					* mueff (float): Variance-effectiveness of sum :math:`w_i x_i`.
					* cc (float): Time constant for cumulation for C.
					* cs (float): T-const for cumulation for sigma control.
					* c1 (float): Learning rate for rank-one update of C.
					* cmu (float): Learning rate for rank-mu update.
					* dumps (float): Damping for sigma.
					* pc (numpy.ndarray): Evolution paths for C.
					* ps (numpy.ndarray): Evolution paths for sigma.
					* B (numpy.ndarray): Definition of coordinate system.
					* D (numpy.ndarray): Diagonal D defines the scale.
					* C (numpy.ndarray): :math:`C^{\frac{1}{2}}.
					* eigeneval (int): Track update of B and D.
					* chiN (float): TODO
		"""
		xb, fxb, d = Algorithm.initPopulation(self, task)
		xmean = self.rand(task.D)
		lambdA = 4 + int(3 * log(task.D)) if self.lambdA is None else self.lambdA
		mu = lambdA // 4 if self.mu is None else self.mu
		# Strategy parameter setting: Selection
		weights = log(mu + 1 / 2) - log(arange(1, mu + 1))
		weights = weights / sum(weights)
		mueff = sum(weights) ** 2 / sum(weights ** 2)
		# Strategy parameter setting: Adaption
		cc = (4 + mueff / task.D) / (task.D + 4 + 2 * mueff / task.D)
		cs = (mueff + 2) / (task.D + mueff + 5)
		c1 = 2 / ((task.D + 1.3) ** 2 + mueff)
		cmu = fmin(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((task.D + 2) ** 2 + mueff))
		damps = 1 + 2 * fmax(0, sqrt((mueff - 1) / (task.D + 1)) - 1) + cs
		# Initialize dynamic (internal) strategy parameters and constants
		pc, ps = full(task.D, .0), full(task.D, .0)
		B, D = eye(task.D, task.D), full(task.D, 1.)
		C = B * diag(D ** 2) * B.T
		invsqrtC = B * diag(D ** -1) * B.T
		chiN = task.D ** 0.5 * (1 - 1 / (4 * task.D) + 1 / (21 * task.D ** 2))
		d.update({
			'xmean': xmean,
			'mu': mu,
			'lambdA': lambdA,
			'sigma': self.sigma0,
			'weights': weights,
			'mueff': mueff,
			'cc': cc,
			'cs': cs,
			'c1': c1,
			'cmu': cmu,
			'damps': damps,
			'pc': pc,
			'ps': ps,
			'B': B,
			'D': D,
			'C': C,
			'invsqrtC': invsqrtC,
			'eigeneval': 0,
			'chiN': chiN,
		})
		return xb, fxb, d

	def runIteration(self, task, pop, fpop, xb, fxb, xmean, mu, lambdA, sigma, weights, mueff, cc, cs, c1, cmu, damps, pc, ps, B, D, C, invsqrtC, eigeneval, chiN, **dparams):
		r"""Core function of CovarianceMatrixAdaptionEvolutionStrategy algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Currnt individual.
			fpop (float): Current individuals function values.
			xb (numpy.ndarray): Global best position.
			fxb (float): Global best positions function/finess value.
			xmean (flaot): Mean over dimensions.
			mu (int): Number of parents.
			lambdA (int): Number of offsprings.
			sigma (float): Coordinate wise standard deviation (step size).
			weights (numpy.ndarray[float]): Array for weighted recombination.
			mueff (float): Variance-effectiveness of sum :math:`w_i x_i`.
			cc (float): Time constant for cumulation for C.
			cs (float): T-const for cumulation for sigma control.
			c1 (float): Learning rate for rank-one update of C.
			cmu (float): Learning rate for rank-mu update.
			dumps (float): Damping for sigma.
			pc (numpy.ndarray): Evolution paths for C.
			ps (numpy.ndarray): Evolution paths for sigma.
			B (numpy.ndarray): Definition of coordinate system.
			D (numpy.ndarray): Diagonal D defines the scale.
			C (numpy.ndarray): :math:`C^{\frac{1}{2}}.
			eigeneval (int): Track update of B and D.
			chiN (float): TODO
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, Dict[str, Any]):
				1. New individual.
				2. New individuals function/fitness value.
				3. New global best position.
				4. New global best positions function/fitness value.
				3. Additional arguments:
					* xmean (flaot): Mean over dimensions.
					* mu (int): Number of parents.
					* lambdA (int): Number of offsprings.
					* sigma (float): Coordinate wise standard deviation (step size).
					* weights (numpy.ndarray[float]): Array for weighted recombination.
					* mueff (float): Variance-effectiveness of sum :math:`w_i x_i`.
					* cc (float): Time constant for cumulation for C.
					* cs (float): T-const for cumulation for sigma control.
					* c1 (float): Learning rate for rank-one update of C.
					* cmu (float): Learning rate for rank-mu update.
					* dumps (float): Damping for sigma.
					* pc (numpy.ndarray): Evolution paths for C.
					* ps (numpy.ndarray): Evolution paths for sigma.
					* B (numpy.ndarray): Definition of coordinate system.
					* D (numpy.ndarray): Diagonal D defines the scale.
					* C (numpy.ndarray): :math:`C^{\frac{1}{2}}.
					* eigeneval (int): Track update of B and D.
					* chiN (float): TODO
		"""
		# Generate labmda offsprings
		arx = asarray([task.repair(xmean + sigma * B.dot(D * self.rand(task.D)), rnd=self.Rand) for _ in range(lambdA)])
		arf = apply_along_axis(task.eval, 1, arx)
		# sort by fitness and compute weighted mean ito xmean
		sindex, xold = argsort(arf), xmean
		xmean = weights.dot(arx[sindex][:mu])
		# Cumulation: Update evolution paths
		ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
		hsig = norm(ps) / sqrt(1 - (1 - cs) ** (2 * task.Evals / lambdA)) / chiN
		pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
		# Adapt covariance matrix C
		artmp = (1 / sigma) * (arx[sindex[:mu]] - xold)
		C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T.dot(diag(weights)).dot(artmp)
		# Adopt step size sigma
		sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1))
		if isnan(sigma) or sigma == inf: sigma = self.sigma0
		# Decomposition of C into B * diag(D**2) * B.conj().T (diagonalizem)
		if task.Evals - eigeneval > lambdA / (c1 + cmu) / task.D / 10:
			eigeneval = task.Evals
			C = triu(C) + triu(C, 1).T
			C[where(isnan(C))] = 1
			D, B = eig(C)
			D, B = D.real, B.real
			D[where(isnan(D))] = 1
			D = sqrt(D)
			invsqrtC = B.dot(diag(D ** -1).dot(B.T))
			invsqrtC[where(invsqrtC == inf)], invsqrtC[where(invsqrtC == -inf)] = 1, -1
		if arf[sindex[0]] < fxb: xb, fxb = arx[sindex[0]], arf[sindex[0]]
		return arx[sindex[0]], arf[sindex[0]], xb, fxb, {
			'xmean': xmean,
			'mu': mu,
			'lambdA': lambdA,
			'sigma': sigma,
			'weights': weights,
			'mueff': mueff,
			'cc': cc,
			'cs': cs,
			'c1': c1,
			'cmu': cmu,
			'damps': damps,
			'pc': pc,
			'ps': ps,
			'B': B,
			'D': D,
			'C': C,
			'invsqrtC': invsqrtC,
			'eigeneval': eigeneval,
			'chiN': chiN,
		}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
