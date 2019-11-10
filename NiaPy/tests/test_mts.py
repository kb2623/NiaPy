# encoding=utf8
from unittest import TestCase

from numpy import random as rnd, apply_along_axis

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.util import fullArray, Task
from NiaPy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS1v1, MTS_LS2, MTS_LS3, MTS_LS3v1

class MTS_LS1TestCase(TestCase):
	r"""Test case for MTS_LS1 function.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.algorithms.other.mts.MTS_LS1`
	"""
	def setUp(self):
		self.t, self.rnd = Task(D=100, benchmark=MyBenchmark()), rnd.RandomState(1234)
		self.x, self.xb = self.rnd.uniform(-3.5, 1.5, self.t.D), self.rnd.uniform(-1, 1, self.t.D)
		self.xf, self.xbf = self.t.eval(self.x), self.t.eval(self.xb)
		self.improve, self.SR = False, fullArray((1, .3, .5, .923), self.t.D)

	def test_run_fine(self):
		r"""Test if running the funciotn works fine."""
		xk, xkf, xb, xbf, improve, grade, sr = MTS_LS1(self.x, self.xf, self.xb, self.xbf, self.improve, self.SR, self.t, rnd=self.rnd)
		self.assertIsNotNone(xk)
		self.assertIsNotNone(sr)
		self.assertIsNotNone(xb)
		self.assertTrue(improve)
		self.assertAlmostEqual(95, grade)
		self.assertAlmostEqual(230.744, xkf, delta=9e-3)
		self.assertAlmostEqual(35.455, xbf, delta=9e-3)

class MTS_LS1v1TestCase(MTS_LS1TestCase):
	r"""Test case for MTS_LS1v1 function.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.algorithms.other.mts.MTS_LS1v1`
	"""
	def test_run_fine(self):
		r"""Test if running the funciotn works fine."""
		xk, xkf, xb, xbf, improve, grade, sr = MTS_LS1v1(self.x, self.xf, self.xb, self.xbf, self.improve, self.SR, self.t, rnd=self.rnd)
		self.assertIsNotNone(xk)
		self.assertIsNotNone(sr)
		self.assertIsNotNone(xb)
		self.assertTrue(improve)
		self.assertAlmostEqual(85, grade)
		self.assertAlmostEqual(235.281, xkf, delta=9e-3)
		self.assertAlmostEqual(35.455, xbf, delta=9e-3)

class MTS_LS2TestCase(MTS_LS1TestCase):
	r"""Test case for MTS_LS2 function.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.algorithms.other.mts.MTS_LS2`
	"""
	def test_run_fine(self):
		r"""Test if running the funciotn works fine."""
		xk, xkf, xb, xbf, improve, grade, sr = MTS_LS2(self.x, self.xf, self.xb, self.xbf, self.improve, self.SR, self.t, rnd=self.rnd)
		self.assertIsNotNone(xk)
		self.assertIsNotNone(sr)
		self.assertIsNotNone(xb)
		self.assertTrue(improve)
		self.assertAlmostEqual(68, grade)
		self.assertAlmostEqual(166.276, xkf, delta=9e-3)
		self.assertAlmostEqual(35.455, xbf, delta=9e-3)

class MTS_LS3TestCase(MTS_LS1TestCase):
	r"""Test case for MTS_LS3 function.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.algorithms.other.mts.MTS_LS3`
	"""
	def test_run_fine(self):
		r"""Test if running the funciotn works fine."""
		xk, xkf, xb, xbf, improve, grade, sr = MTS_LS3(self.x, self.xf, self.xb, self.xbf, self.improve, self.SR, self.t, rnd=self.rnd)
		self.assertIsNotNone(xk)
		self.assertIsNotNone(sr)
		self.assertIsNotNone(xb)
		self.assertTrue(improve)
		self.assertAlmostEqual(64, grade)
		self.assertAlmostEqual(199.305, xkf, delta=9e-3)
		self.assertAlmostEqual(35.455, xbf, delta=9e-3)

class MTS_LS3v1TestCase(MTS_LS1TestCase):
	r"""Test case for MTS_LS3v1 function.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.algorithms.other.mts.MTS_LS3v1`
	"""
	def test_run_fine(self):
		r"""Test if running the funciotn works fine."""
		xk, xkf, xb, xbf, improve, grade, sr = MTS_LS3v1(self.x, self.xf, self.xb, self.xbf, self.improve, self.SR, self.t, rnd=self.rnd)
		self.assertIsNotNone(xk)
		self.assertIsNotNone(sr)
		self.assertIsNotNone(xb)
		self.assertTrue(improve)
		self.assertAlmostEqual(80, grade)
		self.assertAlmostEqual(260.895, xkf, delta=9e-3)
		self.assertAlmostEqual(35.455, xbf, delta=9e-3)

def population_init_test_func(task, NP, rnd, **kwargs):
	r"""Initialize the starting population."""
	pop = rnd.uniform(-.5, .5, (NP, task.D))
	return pop, apply_along_axis(task.eval, 1, pop)

class MTSTestCase(AlgorithmTestCase):
	r"""Test case for MultipleTrajectorySearch algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.other.MultipleTrajectorySearch`
	"""
	def test_algorithm_info(self):
		r"""Test if algorithm info works fine."""
		i = MultipleTrajectorySearch.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters(self):
		r"""Test if type parameters work fine."""
		d = MultipleTrajectorySearch.typeParameters()
		self.assertTrue(d['NoLsTests'](10))
		self.assertTrue(d['NoLsTests'](0))
		self.assertFalse(d['NoLsTests'](-10))
		self.assertTrue(d['NoLs'](10))
		self.assertTrue(d['NoLs'](0))
		self.assertFalse(d['NoLs'](-10))
		self.assertTrue(d['NoLsBest'](10))
		self.assertTrue(d['NoLsBest'](0))
		self.assertFalse(d['NoLsBest'](-10))
		self.assertTrue(d['NoEnabled'](10))
		self.assertFalse(d['NoEnabled'](0))
		self.assertFalse(d['NoEnabled'](-10))

	def test_manual_population_initialization(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = MultipleTrajectorySearch(NP=10, InitPopFunc=population_init_test_func)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		mts_custom = MultipleTrajectorySearch(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = MultipleTrajectorySearch(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		mts_griewank = MultipleTrajectorySearch(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mts_griewankc = MultipleTrajectorySearch(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

class MTSv1TestCase(AlgorithmTestCase):
	r"""Test case for MultipleTrajectorySearchV1 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.other.MultipleTrajectorySearchV1`
	"""
	def test_algorithm_info(self):
		r"""Test if algorithm info works fine."""
		i = MultipleTrajectorySearchV1.algorithmInfo()
		self.assertIsNotNone(i)

	def test_manual_population_initialization(self):
		r"""Test if manual population initialization works fine."""
		t = Task(D=50, benchmark=MyBenchmark())
		a = MultipleTrajectorySearch(NP=10, InitPopFunc=population_init_test_func)
		pop, fpop, d = a.initPopulation(t)
		self.assertIsNotNone(pop)
		self.assertIsNotNone(fpop)
		self.assertIsNotNone(d)
		self.assertFalse(True in (pop < -.5) or True in (pop > .5))

	def test_custom_works_fine(self):
		r"""Test if running the algorithm with coustome benchmark works fine."""
		mts_custom = MultipleTrajectorySearchV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = MultipleTrajectorySearchV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		r"""Test if running the algorithm with benchmark works fine."""
		mts_griewank = MultipleTrajectorySearchV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mts_griewankc = MultipleTrajectorySearchV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
