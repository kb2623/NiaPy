# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import Task
from NiaPy.algorithms.other import HillClimbAlgorithm

def init_population(task, NP, rnd, **kwargs):
	r"""Initialize the populaiton."""
	x = -.5 + rnd.rand(task.D) * 0.75
	return x, task.eval(x)

class HCTestCase(AlgorithmTestCase):
	r"""Test case for HillClimbAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič and Jan Popič

	See Also:
		* :class:`NiaPy.algorithms.other.HillClimbAlgorithm`
	"""
	def test_algorithm_info_fine(self):
		self.assertIsNotNone(HillClimbAlgorithm.algorithmInfo())

	def test_type_parameters_fine(self):
		d = HillClimbAlgorithm.typeParameters()
		self.assertIsNotNone(d.get('delta', None))
		self.assertFalse(d['delta'](-1))
		self.assertFalse(d['delta'](-.1))
		self.assertTrue(d['delta'](1))
		self.assertTrue(d['delta'](.1))

	def test_manual_population_initialization_fine(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = HillClimbAlgorithm(NP=10, InitPopFunc=init_population)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

	def test_custom_works_fine(self):
		ihc_custom = HillClimbAlgorithm(delta=0.4, seed=self.seed)
		ihc_customc = HillClimbAlgorithm(delta=0.4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_custom, ihc_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ihc_griewank = HillClimbAlgorithm(delta=0.1, seed=self.seed)
		ihc_griewankc = HillClimbAlgorithm(delta=0.1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_griewank, ihc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
