# encoding=utf8
import importlib.resources as pkgres
from io import StringIO
from math import pow, isnan
from unittest import TestCase

from numpy import asarray, pi, full, inf, random as rnd, arange, array_equal
import pandas as pd

import NiaPy.data as pkg_data
from NiaPy.util import Utility
from NiaPy.benchmarks import LennardJones, Easom, DeflectedCorrugatedSpring, NeedleEye, Exponential, XinSheYang01, XinSheYang02, XinSheYang03, XinSheYang04, YaoLiu09, Deb01, Deb02, Bohachevsky, Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, StyblinskiTang, Stepint, Step, Step2, Step3, SchumerSteiglitz, Salomon, Quintic, Pinter, Alpine1, Alpine2, ChungReynolds, Csendes, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, ExpandedSchafferF6, SchafferN2, SchafferN4, HGBat, Katsuura, ModifiedSchwefel, Weierstrass, HappyCat, Qing, Ridge, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, Tchebychev, Hilbert, Clustering, ClusteringMin, ClusteringMinPenalty, ClusteringClassification

class TestBenchmarkFunctions(TestCase):
	"""Testing the benchmarks."""
	def setUp(self):
		"""Set up the tests."""
		self.D = 5
		self.array = asarray([0, 0, 0, 0, 0], dtype=float)
		self.array2 = asarray([1, 1, 1, 1, 1], dtype=float)
		self.array3 = asarray([420.968746, 420.968746, 420.968746, 420.968746, 420.968746])
		self.array4 = asarray([-2.903534, -2.903534])
		self.array5 = asarray([-0.5, -0.5, -0.5, -0.5, -0.5])
		self.array6 = asarray([-1, -1, -1, -1, -1], dtype=float)
		self.array7 = asarray([2, 2, 2, 2, 2], dtype=float)
		self.array8 = asarray([7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172])
		self.array9 = asarray([-5.12, -5.12, -5.12, -5.12, -5.12], dtype=float)
		self.array10 = asarray([1, 2, 3, 4, 5], dtype=float)
		self.rand = rnd.RandomState(1)

	def assertBounds(self, bench, lower, upper):
		"""Checking the bounds.

		Arguments:
			bench (Benchmark): Benchmark to test.
			lower (float): Lower bound.
			upper (float): Upper bound.

		Returns:
			Callable[[numpy.ndarray[float]], float: Returns benchmarks evaluation function.
		"""
		b = Utility().get_benchmark(bench)
		self.assertEqual(b.Lower, lower)
		self.assertEqual(b.Upper, upper)
		return b.function()

	def test_rastrigin_latex_code(self):
		self.assertIsNotNone(Rastrigin.latex_code())

	def test_rastrigin_function(self):
		"""Test the rastrigin benchmark."""
		rastrigin = Utility().get_benchmark('rastrigin')
		fun = rastrigin.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array))

	def test_rosenbrock_latex_code(self):
		self.assertIsNotNone(Rosenbrock.latex_code())

	def test_rosenbrock_function(self):
		"""Test the rosenbrock benchmark."""
		rosenbrock = Utility().get_benchmark('rosenbrock')
		fun = rosenbrock.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array2))

	def test_griewank_latex_code(self):
		self.assertIsNotNone(Griewank.latex_code())

	def test_griewank_function(self):
		"""Test the griewank benchmark."""
		griewank = Utility().get_benchmark('griewank')
		fun = griewank.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array))

	def test_sphere_latex_code(self):
		self.assertIsNotNone(Sphere.latex_code())

	def test_sphere_function(self):
		"""Test the sphere benchmark."""
		sphere = Utility().get_benchmark('sphere')
		fun = sphere.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array))

	def test_ackley_latex_code(self):
		self.assertIsNotNone(Ackley.latex_code())

	def test_ackley_function(self):
		"""Test the ackley benchmark."""
		ackley = Utility().get_benchmark('ackley')
		fun = ackley.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.array), places=10)

	def test_schwefel_latex_code(self):
		self.assertIsNotNone(Schwefel.latex_code())

	def test_schwefel_function(self):
		"""Test the schwefel benchmark."""
		schwefel = Utility().get_benchmark('schwefel')
		fun = schwefel.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.array3), places=3)

	def test_schwefel221_latex_code(self):
		self.assertIsNotNone(Schwefel221.latex_code())

	def test_schwefel221_function(self):
		"""Test the schwefel 221 benchmark."""
		schwefel221 = Utility().get_benchmark('schwefel221')
		fun = schwefel221.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array))

	def test_schwefel222_latex_code(self):
		self.assertIsNotNone(Schwefel222.latex_code())

	def test_schwefel222_function(self):
		"""Test the schwefel 222 benchmark."""
		schwefel222 = Utility().get_benchmark('schwefel222')
		fun = schwefel222.function()
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array))

	def test_whitley_latex_code(self):
		self.assertIsNotNone(Whitley.latex_code())

	def test_whitley_function(self):
		"""Test the whitley benchmark."""
		whitley = Utility().get_benchmark('whitley')
		fun = whitley.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.array2))

	def test_styblinskiTang_latex_code(self):
		self.assertIsNotNone(StyblinskiTang.latex_code())

	def test_styblinskiTang_function(self):
		"""Test the styblinski tang benchmark."""
		styblinskiTang = Utility().get_benchmark('styblinskiTang')
		fun = styblinskiTang.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(-78.332, fun(self.array4[:2]), places=3)

	def test_stepint_latex_code(self):
		self.assertIsNotNone(Stepint.latex_code())

	def test_stepint_function(self):
		"""Test the stepint benchmark."""
		stepint = Utility().get_benchmark('stepint')
		fun = stepint.function()
		self.assertTrue(callable(fun))
		self.assertEqual(25.0 - 6 * self.D, fun(self.array9))

	def test_step_latex_code(self):
		self.assertIsNotNone(Step.latex_code())

	def test_step_function(self):
		"""Test the step benchmark."""
		step = Utility().get_benchmark('step')
		fun = step.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_step2_latex_code(self):
		self.assertIsNotNone(Step2.latex_code())

	def test_step2_function(self):
		"""Test the step 2 benchmark."""
		step2 = Utility().get_benchmark('step2')
		fun = step2.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array5), 0.0)

	def test_step3_latex_code(self):
		self.assertIsNotNone(Step3.latex_code())

	def test_step3_function(self):
		"""Test the step3 benchmark."""
		step3 = Utility().get_benchmark('step3')
		fun = step3.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_schumersteiglitz_latex_code(self):
		self.assertIsNotNone(SchumerSteiglitz.latex_code())

	def test_schumerSteiglitz_function(self):
		"""Test the schumer steiglitz benchmark."""
		fun = self.assertBounds('schumerSteiglitz', -100, 100)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_salomon_latex_code(self):
		self.assertIsNotNone(Salomon.latex_code())

	def test_salomon_function(self):
		"""Test the salomon benchmark."""
		fun = self.assertBounds('salomon', -100.0, 100.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_quintic_latex_code(self):
		self.assertIsNotNone(Quintic.latex_code())

	def test_quintic_function(self):
		"""Test the quintic benchmark."""
		fun = self.assertBounds('quintic', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(.0, fun(self.array6))
		self.assertEqual(.0, fun(self.array7))

	def test_pinter_latex_code(self):
		self.assertIsNotNone(Pinter.latex_code())

	def test_pinter_function(self):
		"""Test the pinter benchmark."""
		fun = self.assertBounds('pinter', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_alpine1_latex_code(self):
		self.assertIsNotNone(Alpine1.latex_code())

	def test_alpine1_function(self):
		"""Test the alpine 1 benchmark."""
		fun = self.assertBounds('alpine1', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_alpine2_latex_code(self):
		self.assertIsNotNone(Alpine2.latex_code())

	def test_alpine2_function(self):
		"""Test the apline 2 benchmark."""
		fun = self.assertBounds('alpine2', 0.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(pow(2.8081311800070053291, self.D), fun(self.array8))

	def test_chungReynolds_latex_code(self):
		self.assertIsNotNone(ChungReynolds.latex_code())

	def test_chungReynolds_function(self):
		"""Test the chung reynolds benchmark."""
		fun = self.assertBounds('chungReynolds', -100, 100)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_csendes_latex_code(self):
		self.assertIsNotNone(Csendes.latex_code())

	def test_csendes_function(self):
		"""Test the csendes benchmark."""
		fun = self.assertBounds('csendes', -1.0, 1.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.array), 0.0)

	def test_bentcigar_latex_code(self):
		self.assertIsNotNone(BentCigar.latex_code())

	def test_bentcigar_function(self):
		"""Test the bent cigar benchmark."""
		fun = self.assertBounds('bentcigar', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(0.0, fun(full(2, 0, dtype=float)), delta=1e-4)
		self.assertAlmostEqual(0.0, fun(full(10, 0, dtype=float)), delta=1e-4)
		self.assertAlmostEqual(0.0, fun(full(100, 0, dtype=float)), delta=1e-4)

	def test_discus_latex_code(self):
		self.assertIsNotNone(Discus.latex_code())

	def test_discus_function(self):
		"""Test the discus benchmark."""
		fun = self.assertBounds('discus', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(1000054.0, fun(self.array10))

	def test_elliptic_latex_code(self):
		self.assertIsNotNone(Elliptic.latex_code())

	def test_elliptic_function(self):
		"""Test the elliptic benchmark."""
		fun = self.assertBounds('elliptic', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(self.D, 0.0)), delta=2e6)

	def test_expanded_griewank_plus_rosenbrock_latex_code(self):
		self.assertIsNotNone(ExpandedGriewankPlusRosenbrock.latex_code())

	def test_expanded_griewank_plus_rosenbrock_function(self):
		"""Test the expanded griewank plus rosenbrock benchmark."""
		fun = self.assertBounds('expandedgriewankplusrosenbrock', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(2.2997, fun(self.array), delta=1e2)

	def test_expanded_schaffer_latex_code(self):
		self.assertIsNotNone(ExpandedSchafferF6.latex_code())

	def test_expanded_schaffer_function(self):
		"""Test the expanded schaffer benchmark."""
		fun = self.assertBounds('expandedschaffer', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(2.616740208857464, fun(self.array10), delta=1e-4)

	def test_schaffern2_latex_code(self):
		self.assertIsNotNone(SchafferN2.latex_code())

	def test_schaffern2_function(self):
		"""Test the schaffer n. 2 benchmark."""
		fun = self.assertBounds('schaffer2', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(0.02467, fun(self.array10), delta=1e-4)

	def test_schaffern4_latex_code(self):
		self.assertIsNotNone(SchafferN4.latex_code())

	def test_schaffern4_function(self):
		"""Test the schaffer n. 4 benchmark."""
		fun = self.assertBounds('schaffer4', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(0.97545, fun(self.array10), delta=1e-4)

	def test_hgbat_latex_code(self):
		self.assertIsNotNone(HGBat.latex_code())

	def test_hgbat_function(self):
		"""Test the hgbat benchmark."""
		fun = self.assertBounds('hgbat', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(61.91502622129181, fun(self.array10), delta=60)

	def test_katsuura_latex_code(self):
		self.assertIsNotNone(Katsuura.latex_code())

	def test_katsuura_function(self):
		"""Test the katsuura benchmark."""
		fun = self.assertBounds('katsuura', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(3837.4739882594373, fun(self.array10), delta=4000)

	def test_modifiedscweffel_latex_code(self):
		self.assertIsNotNone(ModifiedSchwefel.latex_code())

	def test_modifiedscwefel_function(self):
		"""Test the modified scwefel benchmark."""
		fun = self.assertBounds('modifiedscwefel', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(6.9448853328785844, fun(self.array10), delta=350)

	def test_weierstrass_latex_code(self):
		self.assertIsNotNone(Weierstrass.latex_code())

	def test_weierstrass_function(self):
		"""Test the weierstrass benchmark."""
		fun = self.assertBounds('weierstrass', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.array10), delta=1e-4)

	def test_happyCat_latex_code(self):
		self.assertIsNotNone(HappyCat.latex_code())

	def test_happyCat_function(self):
		"""Test the happy cat benchmark."""
		fun = self.assertBounds('happyCat', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(11.659147948472494, fun(self.array10))

	def test_qing_latex_code(self):
		self.assertIsNotNone(Qing.latex_code())

	def test_qing_function(self):
		"""Test the quing benchmark."""
		fun = self.assertBounds('qing', -500, 500)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(669.0, fun(self.array10), delta=1e-4)

	def test_ridge_latex_code(self):
		self.assertIsNotNone(Ridge.latex_code())

	def test_ridge_function(self):
		"""Test the ridge benchmark."""
		fun = self.assertBounds('ridge', -64, 64)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(55, fun(full(self.D, -1.0)))
		self.assertAlmostEqual(371.0, fun(self.array10), delta=1e-4)

	def test_michalewicz_latex_code(self):
		self.assertIsNotNone(Michalewichz.latex_code())

	def test_michalewicz_function(self):
		"""Test the michalewicz benchmark."""
		fun = self.assertBounds('michalewicz', 0, pi)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(-1.8013, fun(asarray([2.20, 1.57])), delta=1e-3)

	def test_levy_latex_code(self):
		self.assertIsNotNone(Levy.latex_code())

	def test_levy_function(self):
		"""Test the levy benchmark."""
		fun = self.assertBounds('levy', 0, pi)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(2, 1.)))
		self.assertAlmostEqual(.0, fun(full(10, 1.)))
		self.assertAlmostEqual(.0, fun(full(100, 1.)))

	def test_sphere2_latex_code(self):
		self.assertIsNotNone(Sphere2.latex_code())

	def test_sphere2_function(self):
		"""Test the sphere 2 benchmark."""
		fun = self.assertBounds('sphere2', -1, 1)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(2, 0.)))
		self.assertAlmostEqual(.0, fun(full(10, 0.)))
		self.assertAlmostEqual(.0, fun(full(100, 0.)))

	def test_sphere3_latex_code(self):
		self.assertIsNotNone(Sphere3.latex_code())

	def test_sphere3_function(self):
		"""Test the sphere 3 benchmark."""
		fun = self.assertBounds('sphere3', -65.536, 65.536)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(2, 0.)))
		self.assertAlmostEqual(.0, fun(full(10, 0.)))
		self.assertAlmostEqual(.0, fun(full(100, 0.)))

	def test_trid_latex_code(self):
		self.assertIsNotNone(Trid.latex_code())

	def __trid_opt(self, d):
		"""Trid benchmark optimum."""
		return -d * (d + 4) * (d - 1) / 6

	def __trid_opt_sol(self, d):
		"""Trid optimal solution."""
		return asarray([i * (d + 1 - i) for i in range(1, d + 1)], dtype=float)

	def test_trid_function(self):
		"""Test the trid benchmark."""
		fun = self.assertBounds('trid', -2 ** 2, 2 ** 2)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(self.__trid_opt(2), fun(self.__trid_opt_sol(2)))
		self.assertAlmostEqual(self.__trid_opt(10), fun(self.__trid_opt_sol(10)))
		self.assertAlmostEqual(self.__trid_opt(100), fun(self.__trid_opt_sol(100)))

	def test_perm_latex_code(self):
		self.assertIsNotNone(Perm.latex_code())

	def __perm_opt_sol(self, d):
		"""The perm optimal solution."""
		return asarray([1 / (i + 1) for i in range(d)], dtype=float)

	def test_perm_function(self):
		"""Test the perm bencmark."""
		fun = self.assertBounds('perm', -10, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.__perm_opt_sol(2)))
		self.assertAlmostEqual(.0, fun(self.__perm_opt_sol(10)))
		self.assertAlmostEqual(.0, fun(self.__perm_opt_sol(100)))

	def test_zakharov_latex_code(self):
		self.assertIsNotNone(Zakharov.latex_code())

	def test_zakharov(self):
		"""Test the zakharov benchmark."""
		fun = self.assertBounds('zakharov', -5, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_dixonprice_latex_code(self):
		self.assertIsNotNone(DixonPrice.latex_code())

	def __dixonprice_opt_sol(self, d):
		"""The dixon price optimal solution."""
		return asarray([2 ** (-(2 ** i - 2) / 2 ** i) for i in range(1, d + 1)], dtype=float)

	def test_dixonprice_function(self):
		"""Test the dixon price benchmark."""
		fun = self.assertBounds('dixonprice', -10, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(self.__dixonprice_opt_sol(2)))
		self.assertAlmostEqual(.0, fun(self.__dixonprice_opt_sol(10)))
		self.assertAlmostEqual(.0, fun(self.__dixonprice_opt_sol(100)))

	def test_powell_latex_code(self):
		self.assertIsNotNone(Powell.latex_code())

	def test_powell_function(self):
		"""Tests the powell benchmark."""
		fun = self.assertBounds('powell', -4, 5)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_cosinemixture_latex_code(self):
		self.assertIsNotNone(CosineMixture.latex_code())

	def test_cosinemixture_function(self):
		"""Test the cosine mixture benchmark."""
		fun = self.assertBounds('cosinemixture', -1, 1)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(-.1 * 2, fun(full(2, .0)))
		self.assertAlmostEqual(-.1 * 10, fun(full(10, .0)))
		self.assertAlmostEqual(-.1 * 100, fun(full(100, .0)))

	def test_infinity_latex_code(self):
		self.assertIsNotNone(Infinity.latex_code())

	def test_infinity_function(self):
		"""Test the infinity benchmark."""
		fun = self.assertBounds('infinity', -1, 1)
		self.assertTrue(callable(fun))
		self.assertTrue(isnan(fun(full(2, .0))))
		self.assertTrue(isnan(fun(full(10, .0))))
		self.assertTrue(isnan(fun(full(100, .0))))
		self.assertAlmostEqual(.0, fun(full(2, 1e-4)))
		self.assertAlmostEqual(.0, fun(full(10, 1e-4)))
		self.assertAlmostEqual(.0, fun(full(100, 1e-4)))

	def test_lennard_jones_latex_code(self):
		lc = LennardJones.latex_code()
		self.assertIsNotNone(lc)

	def test_lennard_jones_function(self):
		"""Test the lennard jones benchmark."""
		fun = self.assertBounds('lennardjones', -inf, inf)
		self.assertAlmostEqual(-1.1556391233552166e-05, fun(self.rand.uniform(-10, 10, 9)))

	def test_tchebychev_latex_code(self):
		lc = Tchebychev.latex_code()
		self.assertIsNotNone(lc)

	def test_tchebychev_function(self):
		"""Test the lennard jones benchmark."""
		fun = self.assertBounds('tchebychev', -inf, inf)
		self.assertAlmostEqual(41459.23982180634, fun(self.rand.uniform(-10, 10, 9)))

	def test_hilbert_latex_code(self):
		lc = Hilbert.latex_code()
		self.assertIsNotNone(lc)

	def test_hilbert_function(self):
		"""Test the lennard jones benchmark."""
		fun = self.assertBounds('hilbert', -inf, inf)
		self.assertAlmostEqual(46.01429659846749, fun(self.rand.uniform(-10, 10, 9)))

	def test_easom_function_latex_code(self):
		r"""Test easom latex code."""
		lc = Easom.latex_code()
		self.assertIsNotNone(lc)

	def test_easom_function(self):
		r"""Test easom function."""
		fun = self.assertBounds('easom', -100, 100)
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_deflected_corrugated_spring_function_latex_code(self):
		r"""Test easom latex code."""
		lc = Easom.latex_code()
		self.assertIsNotNone(lc)

	def test_deflected_corrugated_spring_function(self):
		r"""Test easom function."""
		fun = self.assertBounds('deflectedcorrugatedspring', 0, 10)
		DeflectedCorrugatedSpring.alpha = 5.
		self.assertAlmostEqual(-.2, fun(full(2, 5.)))
		self.assertAlmostEqual(-1.0, fun(full(10, 5.)))
		self.assertAlmostEqual(-10.0, fun(full(100, 5.)))

	def test_needle_eye_function_latex_code(self):
		r"""Test easom latex code."""
		lc = NeedleEye.latex_code()
		self.assertIsNotNone(lc)

	def test_needle_eye_function(self):
		r"""Test easom function."""
		fun = self.assertBounds('needleeye', -100, 100)
		NeedleEye.eye = 1.
		self.assertAlmostEqual(420., fun(full(2, 5.)))
		self.assertAlmostEqual(.0, fun(full(2, 1.)))
		self.assertAlmostEqual(10500., fun(full(10, 5.)))
		self.assertAlmostEqual(.0, fun(full(10, 1.)))
		self.assertAlmostEqual(1050000., fun(full(100, 5.)))
		self.assertAlmostEqual(.0, fun(full(100, 1.)))

	def test_exponential_function_latex_code(self):
		r"""Test exponential latex code."""
		lc = Exponential.latex_code()
		self.assertIsNotNone(lc)

	def test_exponential_function(self):
		r"""Test exponential function."""
		fun = self.assertBounds('exponential', -1, 1)
		self.assertAlmostEqual(-1.0, fun(full(2, .0)))
		self.assertAlmostEqual(-1.0, fun(full(10, .0)))
		self.assertAlmostEqual(-1.0, fun(full(100, .0)))

	def test_xin_she_yang_01_function_latex_code(self):
		r"""Test xinsheyang01 latex code."""
		lc = XinSheYang01.latex_code()
		self.assertIsNotNone(lc)

	def test_xin_she_yang_01_function(self):
		r"""Test xinsheyang01 function."""
		fun = self.assertBounds('xinsheyang01', -5, 5)
		fun = XinSheYang01(epsilon=self.rand.rand(100)).function()
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_xin_she_yang_02_function_latex_code(self):
		r"""Test xinsheyang02 latex code."""
		lc = XinSheYang02.latex_code()
		self.assertIsNotNone(lc)

	def test_xin_she_yang_02_function(self):
		r"""Test xinsheyang02 function."""
		fun = self.assertBounds('xinsheyang02', -2 * pi, 2 * pi)
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_xin_she_yang_03_function_latex_code(self):
		r"""Test xinsheyang03 latex code."""
		lc = XinSheYang03.latex_code()
		self.assertIsNotNone(lc)

	def test_xin_she_yang_03_function(self):
		r"""Test xinsheyang03 function."""
		fun = self.assertBounds('xinsheyang03', -20, 20)
		self.assertAlmostEqual(-1.0, fun(full(2, .0)))
		self.assertAlmostEqual(-1.0, fun(full(10, .0)))
		self.assertAlmostEqual(-1.0, fun(full(100, .0)))

	def test_xin_she_yang_04_function_latex_code(self):
		r"""Test xinsheyang04 latex code."""
		lc = XinSheYang04.latex_code()
		self.assertIsNotNone(lc)

	def test_xin_she_yang_04_function(self):
		r"""Test xinsheyang04 function."""
		fun = self.assertBounds('xinsheyang04', -10, 10)
		self.assertAlmostEqual(-1.0, fun(full(2, .0)))
		self.assertAlmostEqual(-1.0, fun(full(10, .0)))
		self.assertAlmostEqual(-1.0, fun(full(100, .0)))

	def test_yao_liu_09_function_latex_code(self):
		r"""Test xinsheyang04 latex code."""
		lc = YaoLiu09.latex_code()
		self.assertIsNotNone(lc)

	def test_yao_liu_09_function(self):
		r"""Test xinsheyang04 function."""
		fun = self.assertBounds('yaoliu09', -5.12, 5.12)
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_deb_01_function_latex_code(self):
		r"""Test deb01 latex code."""
		lc = Deb01.latex_code()
		self.assertIsNotNone(lc)

	def test_deb_01_function(self):
		r"""Test deb01 function."""
		fun = self.assertBounds('deb01', -1, 1)
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_deb_02_function_latex_code(self):
		r"""Test dob02 latex code."""
		lc = Deb02.latex_code()
		self.assertIsNotNone(lc)

	def test_deb_02_function(self):
		r"""Test deb02 function."""
		fun = self.assertBounds('deb02', 0, 1)
		x_opt = 0.0184201574932019
		self.assertAlmostEqual(.0, fun(full(2, x_opt)))
		self.assertAlmostEqual(.0, fun(full(10, x_opt)))
		self.assertAlmostEqual(.0, fun(full(100, x_opt)))

	def test_bohachevsky_function_latex_code(self):
		r"""Test bohachevsky latex code."""
		lc = Bohachevsky.latex_code()
		self.assertIsNotNone(lc)

	def test_bohachevsky_function(self):
		r"""Test bohachevsky function."""
		fun = self.assertBounds('bohachevsky', -15, 15)
		self.assertAlmostEqual(.0, fun(full(2, .0)))
		self.assertAlmostEqual(.0, fun(full(10, .0)))
		self.assertAlmostEqual(.0, fun(full(100, .0)))

	def test_clustering_function(self):
		r"""Test clustering function."""
		df = pd.read_csv(StringIO(pkgres.read_text(pkg_data, 'glass.csv')))
		clustering_problem = Clustering(df.iloc[:, :-1].values)
		fun = clustering_problem.function()
		self.assertTrue(array_equal(asarray([1.511150, 10.730000, .0, .29, 69.81, .0, 5.43, .0, .0]), clustering_problem.Lower))
		self.assertTrue(array_equal(asarray([1.533930, 17.38, 4.49, 3.5, 75.41, 6.21, 16.19, 3.15, .51]), clustering_problem.Upper))
		self.assertAlmostEqual(1919.551062139, fun(arange(3 * 9).astype(float)))

	def test_clustering_min_function(self):
		r"""Test clustering min function."""
		df = pd.read_csv(StringIO(pkgres.read_text(pkg_data, 'glass.csv')))
		clustering_problem = ClusteringMin(df.iloc[:, :-1].values)
		fun = clustering_problem.function()
		self.assertTrue(array_equal(asarray([1.511150, 10.730000, .0, .29, 69.81, .0, 5.43, .0, .0]), clustering_problem.Lower))
		self.assertTrue(array_equal(asarray([1.533930, 17.38, 4.49, 3.5, 75.41, 6.21, 16.19, 3.15, .51]), clustering_problem.Upper))
		self.assertAlmostEqual(601.3084140480892, fun(arange(3 * 9).astype(float)))

	def test_clustering_min_penalty_function(self):
		r"""Test clustering min with penalty function."""
		df = pd.read_csv(StringIO(pkgres.read_text(pkg_data, 'glass.csv')))
		clustering_problem = ClusteringMinPenalty(df.iloc[:, :-1].values)
		fun = clustering_problem.function()
		self.assertTrue(array_equal(asarray([1.511150, 10.730000, .0, .29, 69.81, .0, 5.43, .0, .0]), clustering_problem.Lower))
		self.assertTrue(array_equal(asarray([1.533930, 17.38, 4.49, 3.5, 75.41, 6.21, 16.19, 3.15, .51]), clustering_problem.Upper))
		self.assertAlmostEqual(601.3084140480892, fun(arange(3 * 9).astype(float)))
		self.assertAlmostEqual(698.254442793479, fun(full(3 * 9, .5).astype(float)))

	def test_clustering_classification_function(self):
		r"""Test clustering min with penalty based on classification function."""
		df = pd.read_csv(StringIO(pkgres.read_text(pkg_data, 'glass.csv')))
		clustering_problem = ClusteringClassification(df.iloc[:, :-1].values, df.iloc[:, -1].values)
		fun = clustering_problem.function()
		self.assertTrue(array_equal(asarray([1.511150, 10.730000, .0, .29, 69.81, .0, 5.43, .0, .0]), clustering_problem.Lower))
		self.assertTrue(array_equal(asarray([1.533930, 17.38, 4.49, 3.5, 75.41, 6.21, 16.19, 3.15, .51]), clustering_problem.Upper))
		self.assertAlmostEqual(676.0630650761266, fun(arange(3 * 9).astype(float)))
		self.assertAlmostEqual(733.9241747560959, fun(full(3 * 9, .5).astype(float)))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
