# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import Task
from NiaPy.algorithms.other import SimulatedAnnealing
from NiaPy.algorithms.other.sa import coolLinear

def init_population(task, NP, rnd, **kwargs):
	r"""Initialize the starting population."""
	x = -.5 + rnd.rand(task.D) * 0.75
	return x, task.eval(x)

class SATestCase(AlgorithmTestCase):
	r"""Test case for SimulatedAnnealing algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.other.SimulatedAnnealing`
	"""
	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		i = SimulatedAnnealing.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		r"""Test if type parameters work fine."""
		d = SimulatedAnnealing.typeParameters()
		self.assertTrue(d['delta'](1))
		self.assertFalse(d['delta'](0))
		self.assertFalse(d['delta'](-1))
		self.assertTrue(d['T'](1))
		self.assertFalse(d['T'](0))
		self.assertFalse(d['T'](-1))
		self.assertTrue(d['deltaT'](1))
		self.assertFalse(d['deltaT'](0))
		self.assertFalse(d['deltaT'](-1))
		self.assertTrue(d['epsilon'](0.1))
		self.assertFalse(d['epsilon'](-0.1))
		self.assertFalse(d['epsilon'](10))

	def test_manual_population_initialization_fine(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = SimulatedAnnealing(NP=10, InitPopFunc=init_population)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

	def test_custom_works_fine(self):
		r"""Test if manual population initialization works fine."""
		ca_custom = SimulatedAnnealing(seed=self.seed)
		ca_customc = SimulatedAnnealing(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		ca_griewank = SimulatedAnnealing(seed=self.seed)
		ca_griewankc = SimulatedAnnealing(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

	def test_custom1_works_fine(self):
		r"""Test if manual population initialization works fine."""
		ca_custom = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		ca_customc = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank1_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		ca_griewank = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		ca_griewankc = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
