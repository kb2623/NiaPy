# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CuckooSearch

class CSTestCase(AlgorithmTestCase):
	def test_basic_information_fine(self):
		i = CuckooSearch.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters_fine(self):
		d = CuckooSearch.typeParameters()
		self.assertIsNotNone(d)
		# Test for N parameter checks
		self.assertIsNotNone(d.get('N', None))
		self.assertFalse(d['N'](-1))
		self.assertFalse(d['N'](0))
		self.assertFalse(d['N'](.3))
		self.assertTrue(d['N'](1))
		self.assertTrue(d['N'](100))
		# Test for pa parameter checks
		self.assertIsNotNone(d.get('pa', None))
		self.assertFalse(d['pa'](-.3))
		self.assertFalse(d['pa'](-10))
		self.assertFalse(d['pa'](10))
		self.assertTrue(d['pa'](.0))
		self.assertTrue(d['pa'](.3))
		self.assertTrue(d['pa'](1.))
		# Test for alpha parameter checks
		self.assertIsNotNone(d.get('alpha', None))
		self.assertTrue(d['alpha'](-.1))
		self.assertTrue(d['alpha'](-10))
		self.assertTrue(d['alpha'](10))
		self.assertTrue(d['alpha'](.1))

	def test_custom_works_fine(self):
		cs_custom = CuckooSearch(N=20, seed=self.seed)
		cs_customc = CuckooSearch(N=20, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_custom, cs_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		cs_griewank = CuckooSearch(N=10, seed=self.seed)
		cs_griewankc = CuckooSearch(N=10, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_griewank, cs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
