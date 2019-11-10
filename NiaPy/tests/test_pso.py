# encoding=utf8
from NiaPy.algorithms.basic import ParticleSwarmOptimization, ParticleSwarmAlgorithm, OppositionVelocityClampingParticleSwarmOptimization, CenterParticleSwarmOptimization, MutatedParticleSwarmOptimization, MutatedCenterParticleSwarmOptimization, ComprehensiveLearningParticleSwarmOptimizer, MutatedCenterUnifiedParticleSwarmOptimization
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class PSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = ParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = ParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))

	def test_custom_works_fine(self):
		pso_custom = ParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		pso_customc = ParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, pso_custom, pso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		pso_griewank = ParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		pso_griewankc = ParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, pso_griewank, pso_griewankc)

class WVCPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = ParticleSwarmAlgorithm.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = ParticleSwarmAlgorithm.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, wvcpso_custom, wvcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		wvcpso_griewank = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_griewankc = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, wvcpso_griewank, wvcpso_griewankc)

class OVCPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = OppositionVelocityClampingParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = OppositionVelocityClampingParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		wvcpso_custom = OppositionVelocityClampingParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_customc = OppositionVelocityClampingParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, wvcpso_custom, wvcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		wvcpso_griewank = OppositionVelocityClampingParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		wvcpso_griewankc = OppositionVelocityClampingParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, wvcpso_griewank, wvcpso_griewankc)

class CPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = CenterParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = CenterParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		cpso_custom = CenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		cpso_customc = CenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cpso_custom, cpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		cpso_griewank = CenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		cpso_griewankc = CenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cpso_griewank, cpso_griewankc)

class MPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = MutatedParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = MutatedParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mpso_custom = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mpso_customc = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mpso_custom, mpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mpso_griewank = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mpso_griewankc = MutatedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mpso_griewank, mpso_griewankc)

class MCPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = MutatedCenterParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = MutatedCenterParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcpso_custom = MutatedCenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcpso_customc = MutatedCenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mcpso_custom, mcpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mcpso_griewank = MutatedCenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcpso_griewankc = MutatedCenterParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mcpso_griewank, mcpso_griewankc)

class MCUPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = MutatedCenterUnifiedParticleSwarmOptimization.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = MutatedCenterUnifiedParticleSwarmOptimization.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['vMin'](None))
		self.assertFalse(d['vMax'](None))
		self.assertFalse(d['w'](None))
		self.assertFalse(d['w'](-.1))
		self.assertFalse(d['w'](-10))
		self.assertTrue(d['w'](.01))
		self.assertTrue(d['w'](10.01))

	def test_custom_works_fine(self):
		mcupso_custom = MutatedCenterUnifiedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcupso_customc = MutatedCenterUnifiedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mcupso_custom, mcupso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mcupso_griewank = MutatedCenterUnifiedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		mcupso_griewankc = MutatedCenterUnifiedParticleSwarmOptimization(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mcupso_griewank, mcupso_griewankc)

class CLPSOTestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		al = ComprehensiveLearningParticleSwarmOptimizer.algorithmInfo()
		self.assertIsNotNone(al)

	def test_parameter_type(self):
		d = ComprehensiveLearningParticleSwarmOptimizer.typeParameters()
		self.assertTrue(d['C1'](10))
		self.assertTrue(d['C2'](10))
		self.assertTrue(d['C1'](0))
		self.assertTrue(d['C2'](0))
		self.assertFalse(d['C1'](-10))
		self.assertFalse(d['C2'](-10))
		self.assertTrue(d['vMax'](10))
		self.assertTrue(d['vMin'](10))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))

	def test_custom_works_fine(self):
		clpso_custom = ComprehensiveLearningParticleSwarmOptimizer(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		clpso_customc = ComprehensiveLearningParticleSwarmOptimizer(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, clpso_custom, clpso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		clpso_griewank = ComprehensiveLearningParticleSwarmOptimizer(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		clpso_griewankc = ComprehensiveLearningParticleSwarmOptimizer(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, clpso_griewank, clpso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
