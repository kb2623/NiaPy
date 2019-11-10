# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import Task
from NiaPy.algorithms.basic import BlackHole

def init_population(task, NP, rnd, **kwargs):
	r"""Initialize the starting population."""
	x = -.5 + rnd.rand(task.D) * 0.75
	return x, task.eval(x)

class BHTestCase(AlgorithmTestCase):
	r"""Test case for Black Hole algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.other.SimulatedAnnealing`
	"""
	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		i = BlackHole.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		r"""Test if type parameters work fine."""
		d = BlackHole.typeParameters()
		self.assertTrue(d['NP'](1))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-1))

	def test_manual_population_initialization_fine(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = BlackHole(NP=50, InitPopFunc=init_population)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

	def test_custom_works_fine(self):
		r"""Test if manual population initialization works fine."""
		bh_custom = BlackHole(NP=50, seed=self.seed)
		bh_customc = BlackHole(NP=50, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bh_custom, bh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		bh_griewank = BlackHole(NP=50, seed=self.seed)
		bh_griewankc = BlackHole(NP=50, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bh_griewank, bh_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
