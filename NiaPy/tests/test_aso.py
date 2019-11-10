# encoding=utf8

from unittest import TestCase

from numpy import apply_along_axis

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import Task
from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.algorithms.other.aso import Elitism, Sequential, Crossover

def population_init_test_func(task, NP, rnd, **kwargs):
	r"""Initialize population."""
	pop = rnd.uniform(-.5, .5, (NP, task.D))
	return pop, apply_along_axis(task.eval, 1, pop)

class ASOTestCase(TestCase):
	r"""Test case for AnarchicSocietyOptimization algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithm.other.AnarchicSocietyOptimization`
		* :func:`NiaPy.algorithm.other.aso.Elitism`
	"""
	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		i = AnarchicSocietyOptimization.algorithmInfo()
		self.assertIsNotNone(i)

	def test_parameter_types(self):
		r"""Test if type parameters work fine."""
		d = AnarchicSocietyOptimization.typeParameters()
		self.assertTrue(d['NP'](1))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-1))
		self.assertTrue(d['F'](10))
		self.assertFalse(d['F'](0))
		self.assertFalse(d['F'](-10))
		self.assertTrue(d['CR'](0.1))
		self.assertFalse(d['CR'](-19))
		self.assertFalse(d['CR'](19))
		def params_test(s):
			self.assertFalse(d[s](10))
			self.assertFalse(d[s](10.))
			self.assertFalse(d[s](-10))
			self.assertFalse(d[s](-10.))
			self.assertTrue(d[s](1.))
			self.assertTrue(d[s](.1))
			self.assertFalse(d[s](-1.))
			self.assertFalse(d[s](-.1))
		# Test alpha parameter
		params_test('alpha')
		# Test alpha parameter
		params_test('gamma')
		# Test alpha parameter
		params_test('theta')

	def test_manual_population_initialization(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = AnarchicSocietyOptimization(NP=10, InitPopFunc=population_init_test_func)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

class ASOElitismTestCase(AlgorithmTestCase):
	r"""Test case for AnarchicSocietyOptimization algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithm.other.AnarchicSocietyOptimization`
		* :func:`NiaPy.algorithm.other.aso.Elitism`
	"""
	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

class ASOSequentialTestCase(AlgorithmTestCase):
	r"""Test case for AnarchicSocietyOptimization algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithm.other.AnarchicSocietyOptimization`
		* :func:`NiaPy.algorithm.other.aso.Sequential`
	"""
	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

class ASOCrossoverTestCase(AlgorithmTestCase):
	r"""Test case for AnarchicSocietyOptimization algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithm.other.AnarchicSocietyOptimization`
		* :func:`NiaPy.algorithm.other.aso.Crossover`
	"""
	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
