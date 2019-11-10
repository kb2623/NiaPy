"""Module with implementations of benchmark functions."""

from NiaPy.benchmarks.benchmark import Benchmark
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.benchmarks.alpine import Alpine1, Alpine2
from NiaPy.benchmarks.rastrigin import Rastrigin
from NiaPy.benchmarks.rosenbrock import Rosenbrock
from NiaPy.benchmarks.griewank import Griewank, ExpandedGriewankPlusRosenbrock
from NiaPy.benchmarks.schwefel import Schwefel, Schwefel221, Schwefel222, ModifiedSchwefel
from NiaPy.benchmarks.whitley import Whitley
from NiaPy.benchmarks.happyCat import HappyCat
from NiaPy.benchmarks.ridge import Ridge
from NiaPy.benchmarks.chungReynolds import ChungReynolds
from NiaPy.benchmarks.csendes import Csendes
from NiaPy.benchmarks.pinter import Pinter
from NiaPy.benchmarks.qing import Qing
from NiaPy.benchmarks.quintic import Quintic
from NiaPy.benchmarks.salomon import Salomon
from NiaPy.benchmarks.schumerSteiglitz import SchumerSteiglitz
from NiaPy.benchmarks.step import Step, Step2, Step3
from NiaPy.benchmarks.stepint import Stepint
from NiaPy.benchmarks.styblinskiTang import StyblinskiTang
from NiaPy.benchmarks.bentcigar import BentCigar
from NiaPy.benchmarks.weierstrass import Weierstrass
from NiaPy.benchmarks.hgbat import HGBat
from NiaPy.benchmarks.katsuura import Katsuura
from NiaPy.benchmarks.elliptic import Elliptic
from NiaPy.benchmarks.discus import Discus
from NiaPy.benchmarks.michalewichz import Michalewichz
from NiaPy.benchmarks.levy import Levy
from NiaPy.benchmarks.sphere import Sphere, Sphere2, Sphere3
from NiaPy.benchmarks.trid import Trid
from NiaPy.benchmarks.perm import Perm
from NiaPy.benchmarks.zakharov import Zakharov
from NiaPy.benchmarks.dixonprice import DixonPrice
from NiaPy.benchmarks.powell import Powell
from NiaPy.benchmarks.cosinemixture import CosineMixture
from NiaPy.benchmarks.infinity import Infinity
from NiaPy.benchmarks.schaffer import SchafferN2, SchafferN4, ExpandedSchafferF6
from NiaPy.benchmarks.lennardjones import LennardJones
from NiaPy.benchmarks.hilbert import Hilbert
from NiaPy.benchmarks.tchebyshev import Tchebychev
from NiaPy.benchmarks.easom import Easom
from NiaPy.benchmarks.deflectedcorrugatespring import DeflectedCorrugatedSpring
from NiaPy.benchmarks.needleeye import NeedleEye
from NiaPy.benchmarks.exponential import Exponential
from NiaPy.benchmarks.xinsheyang import XinSheYang01, XinSheYang02, XinSheYang03, XinSheYang04
from NiaPy.benchmarks.yaoliu import YaoLiu09
from NiaPy.benchmarks.deb import Deb01, Deb02
from NiaPy.benchmarks.bohachevsky import Bohachevsky
from NiaPy.benchmarks.clustering import Clustering, ClusteringMin, ClusteringMinPenalty, ClusteringClassification

__all__ = [
	'Benchmark',
	'Rastrigin',
	'Rosenbrock',
	'Griewank',
	'ExpandedGriewankPlusRosenbrock',
	'Sphere',
	'Ackley',
	'Schwefel',
	'Schwefel221',
	'Schwefel222',
	'ModifiedSchwefel',
	'Whitley',
	'Alpine1',
	'Alpine2',
	'HappyCat',
	'Ridge',
	'ChungReynolds',
	'Csendes',
	'Pinter',
	'Qing',
	'Quintic',
	'Salomon',
	'SchumerSteiglitz',
	'Step',
	'Step2',
	'Step3',
	'Stepint',
	'StyblinskiTang',
	'BentCigar',
	'Weierstrass',
	'HGBat',
	'Katsuura',
	'Elliptic',
	'Discus',
	'Michalewichz',
	'Levy',
	'Sphere',
	'Sphere2',
	'Sphere3',
	'Trid',
	'Perm',
	'Zakharov',
	'DixonPrice',
	'Powell',
	'CosineMixture',
	'Infinity',
	'ExpandedSchafferF6',
	'SchafferN2',
	'SchafferN4',
	'LennardJones',
	'Hilbert',
	'Tchebychev',
	'Easom',
	'DeflectedCorrugatedSpring',
	'NeedleEye',
	'Exponential',
	'XinSheYang01',
	'XinSheYang02',
	'XinSheYang03',
	'XinSheYang04',
	'YaoLiu09',
	'Deb01',
	'Deb02',
	'Bohachevsky',
	'Clustering',
	'ClusteringMin',
	'ClusteringMinPenalty',
	'ClusteringClassification'
]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
