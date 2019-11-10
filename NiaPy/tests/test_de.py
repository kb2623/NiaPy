# encoding=utf8
from numpy import asarray

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import objects2array, Task
from NiaPy.algorithms.basic import DifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution, AgingNpMultiMutationDifferentialEvolution
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1, proportional, linear, bilinear

def population_init_test_func(task, NP, rnd, itype, **kwargs):
	r"""Initialize pipulation."""
	pop = objects2array([itype(x=rnd.uniform(-0.5, 0.5, task.D), task=task, e=True) for _ in range(NP)])
	return pop, asarray([i.f for i in pop])

class DETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		i = DifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters_fine(self):
		r"""Test if type parameters work fine."""
		d = DifferentialEvolution.typeParameters()
		self.assertIsNotNone(d)
		# Test for NP parameter checks
		self.assertIsNotNone(d.get('NP', None))
		self.assertFalse(d['NP'](.3))
		self.assertFalse(d['NP'](-.3))
		self.assertFalse(d['NP'](-30))
		self.assertTrue(d['NP'](1))
		self.assertTrue(d['NP'](100))
		# Test for F parameter checks
		self.assertIsNotNone(d.get('F', None))
		self.assertFalse(d['F'](-2))
		self.assertFalse(d['F'](-.2))
		self.assertFalse(d['F'](.0))
		self.assertFalse(d['F'](2.034))
		self.assertTrue(d['F'](.1))
		self.assertTrue(d['F'](1))
		self.assertTrue(d['F'](2))
		# Test for CR parameter checks
		self.assertIsNotNone(d.get('CR', None))
		self.assertFalse(d['CR'](-.1))
		self.assertFalse(d['CR'](1.3))
		self.assertTrue(d['CR'](1.))
		self.assertTrue(d['CR'](.1))
		self.assertTrue(d['CR'](.0))

	def test_manual_population_initialization(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = DifferentialEvolution(NP=10, InitPopFunc=population_init_test_func)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		map(lambda e: self.assertFalse(True in (e.x < -.5) or True in (e.x > .5)), pop)

	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		de_custom = DifferentialEvolution(F=0.5, CR=0.9, seed=self.seed)
		de_customc = DifferentialEvolution(F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		de_griewank = DifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

	def test_cross_rand1(self):
		de_rand1 = DifferentialEvolution(CrossMutt=CrossRand1, seed=self.seed)
		de_rand1c = DifferentialEvolution(CrossMutt=CrossRand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand1, de_rand1c)

	def test_cross_best1(self):
		de_best1 = DifferentialEvolution(CrossMutt=CrossBest1, seed=self.seed)
		de_best1c = DifferentialEvolution(CrossMutt=CrossBest1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best1, de_best1c)

	def test_cross_rand2(self):
		de_rand2 = DifferentialEvolution(CrossMutt=CrossRand2, seed=self.seed)
		de_rand2c = DifferentialEvolution(CrossMutt=CrossRand2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand2, de_rand2c)

	def test_cross_best2(self):
		de_best2 = DifferentialEvolution(CrossMutt=CrossBest2, seed=self.seed)
		de_best2c = DifferentialEvolution(CrossMutt=CrossBest2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best2, de_best2c)

	def test_cross_curr2rand1(self):
		de_curr2rand1 = DifferentialEvolution(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		de_curr2rand1c = DifferentialEvolution(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_curr2rand1, de_curr2rand1c)

	def test_cross_curr2best1(self):
		de_curr2best1 = DifferentialEvolution(CrossMutt=CrossCurr2Best1, seed=self.seed)
		de_curr2best1c = DifferentialEvolution(CrossMutt=CrossCurr2Best1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_curr2best1, de_curr2best1c)

class dynNpDETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.basic.DynNpDifferentialEvolution`
	"""
	def test_algorithm_info(self):
		i = DynNpDifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		d = DynNpDifferentialEvolution.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_custom_works_fine(self):
		de_custom = DynNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = DynNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = DynNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DynNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class ANpDETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.AgingNpDifferentialEvolution`
	"""
	def test_algorithm_info(self):
		i = AgingNpDifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		d = AgingNpDifferentialEvolution.typeParameters()
		self.assertIsNotNone(d.pop('Lt_min', None))
		self.assertIsNotNone(d.pop('Lt_max', None))
		self.assertIsNotNone(d.pop('delta_np', None))
		self.assertIsNotNone(d.pop('omega', None))

	def test_custom_works_linear_fine(self):
		de_custom = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=linear, seed=self.seed)
		de_customc = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=linear, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_linear_fine(self):
		de_griewank = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=linear, seed=self.seed)
		de_griewankc = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=linear, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

	def test_custom_works_bilinear_fine(self):
		de_custom = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=bilinear, seed=self.seed)
		de_customc = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=bilinear, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_bilinear_fine(self):
		de_griewank = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=bilinear, seed=self.seed)
		de_griewankc = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=bilinear, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

	def test_custom_works_proportional_fine(self):
		de_custom = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=proportional, seed=self.seed)
		de_customc = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, age=proportional, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_proportional_fine(self):
		de_griewank = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=proportional, seed=self.seed)
		de_griewankc = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, age=proportional, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class MsDETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution`
	"""
	def test_algorithm_info(self):
		i = MultiStrategyDifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		d = MultiStrategyDifferentialEvolution.typeParameters()
		self.assertIsNone(d.get('CrossMutt', None))
		self.assertIsNotNone(d.get('strategies', None))

	def test_Custom_works_fine(self):
		de_custom = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class DynNpMsDETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.basic.DynNpMultiStrategyDifferentialEvolution`
	"""
	def test_algorithm_info(self):
		i = DynNpMultiStrategyDifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		d = DynNpMultiStrategyDifferentialEvolution.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_custom_works_fine(self):
		de_custom = DynNpMultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = DynNpMultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = DynNpMultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DynNpMultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class ANpMsDETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.AgingNpMultiMutationDifferentialEvolution`
	"""
	def test_algorithm_info(self):
		i = AgingNpMultiMutationDifferentialEvolution.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		d = AgingNpMultiMutationDifferentialEvolution.typeParameters()
		self.assertIsNotNone(d.pop('Lt_min', None))
		self.assertIsNotNone(d.pop('Lt_max', None))
		self.assertIsNotNone(d.pop('delta_np', None))
		self.assertIsNotNone(d.pop('omega', None))

	def test_custom_works_fine(self):
		de_custom = AgingNpMultiMutationDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = AgingNpMultiMutationDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = AgingNpMultiMutationDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = AgingNpMultiMutationDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
