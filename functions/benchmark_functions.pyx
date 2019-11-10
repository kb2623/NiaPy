import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "numpy/arrayobject.h":
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "bfuncs.h":
	double ackley_func (double*, int)
	double sphere_func (double*, int)
	double ellips_func (double*, int)
	double bent_cigar_func (double*, int)
	double dixon_price_func (double*, int)
	double griewank_func (double*, int)
	double happycat_func (double*, int)
	double hgbat_func (double*, int)
	double katsuura_func (double*, int)
	double levy_func (double*, int)
	double rastrigin_func (double*, int)
	double rosenbrock_func (double*, int)
	double weierstrass_func (double*, int)
	double Lennard_Jones(double*, int)
	double Hilbert(double*, int)
	double Tchebyshev(double*, int)
	double modified_schwefel_func (double*, int)
	double expanded_scaffer6_func (double*, int)
	double zakharov_func (double*, int)
	double discus_func (double*, int)
	double hgbat_func (double*, int)
	double grie_rosen_func (double*, int)
	double alpine1_func (double*, int)
	double alpine2_func (double*, int)
	double chungreynolds_func (double*, int)
	double cosinemixture_func (double*, int)
	double csendes_func (double*, int)
	double expanded_griewank_plus_rosenbrock_func (double*, int)
	double infinity_func (double*, int)
	double michalewichz_func (double*, int, double)
	double perm_func (double*, int, double)
	double pinter_func (double*, int)
	double powell_func (double*, int)
	double qing_func (double*, int)
	double quintic_func (double*, int)
	double ridge_func (double*, int)
	double salomon_func (double*, int)
	double sphere2_func (double*, int)
	double sphere3_func (double*, int)
	double schwefel_func (double*, int)
	double schwefel1222_func (double*, int)
	double schwefel1221_func (double*, int)
	double step_func (double*, int)
	double step2_func (double*, int)
	double step3_func (double*, int)
	double stepint_func (double*, int)
	double styblinskitang_func (double*, int)
	double trid_func (double*, int)
	double whitley_func (double*, int)
	double schumer_steiglitz_func (double*, int)
	double schaffern2_func (double*)
	double schaffern4_func (double*)
	double easom_func(double*, int, double, double, double)
	double deflected_corrugated_spring_func (double*, int, double, double)
	double needle_eye_func (double*, int, double)
	double bohachevsky_func (double*, int)
	double deb01_func (double*, int)
	double deb02_func (double*, int)
	double exponential_func (double*, int)
	double xin_she_yang_01_func (double*, int, double*)
	double xin_she_yang_02_func (double*, int)
	double xin_she_yang_03_func (double*, int, double, double)
	double xin_she_yang_04_func (double*, int)
	double yaoliu_09_func (double*, int)
	double clustering_func (int, double*, int, double*, int, double*)
	double clustering_min_func (int, double*, int, double*, int, double*)

cpdef double ackley_function(np.ndarray[double, ndim=1, mode='c'] x):
	return ackley_func(&x[0], len(x))

cpdef double sphere_function(np.ndarray[double, ndim=1, mode='c'] x):
	return sphere_func(&x[0], len(x))

cpdef double bent_cigar_function(np.ndarray[double, ndim=1, mode='c'] x):
	return bent_cigar_func(&x[0], len(x))

cpdef double dixon_price_function(np.ndarray[double, ndim=1, mode='c'] x):
	return dixon_price_func(&x[0], len(x))

cpdef double ellips_function(np.ndarray[double, ndim=1, mode='c'] x):
	return ellips_func(&x[0], len(x))

cpdef double griewank_function(np.ndarray[double, ndim=1, mode='c'] x):
	return griewank_func(&x[0], len(x))

cpdef double happycat_function(np.ndarray[double, ndim=1, mode='c'] x):
	return happycat_func(&x[0], len(x))

cpdef double hgbat_function(np.ndarray[double, ndim=1, mode='c'] x):
	return hgbat_func(&x[0], len(x))

cpdef double katsuura_function(np.ndarray[double, ndim=1, mode='c'] x):
	return katsuura_func(&x[0], len(x))

cpdef double levy_function(np.ndarray[double, ndim=1, mode='c'] x):
	return levy_func(&x[0], len(x))

cpdef double rastrigin_function(np.ndarray[double, ndim=1, mode='c'] x):
	return rastrigin_func(&x[0], len(x))

cpdef double rosenbrock_function(np.ndarray[double, ndim=1, mode='c'] x):
	return rosenbrock_func(&x[0], len(x))

cpdef double weierstrass_function(np.ndarray[double, ndim=1, mode='c'] x):
	return weierstrass_func(&x[0], len(x))

cpdef double Lennard_Jones_function(np.ndarray[double, ndim=1, mode='c'] x):
	if len(x) % 3 != 0: return np.inf
	return Lennard_Jones(&x[0], len(x))

cpdef double Hilbert_function(np.ndarray[double, ndim=1, mode='c'] x):
	return Hilbert(&x[0], len(x))

cpdef double Tchebyshev_function(np.ndarray[double, ndim=1, mode='c'] x):
	return Tchebyshev(&x[0], len(x))

cpdef double modified_schwefel_function(np.ndarray[double, ndim=1, mode='c'] x):
	return modified_schwefel_func(&x[0], len(x))

cpdef double expanded_scaffer6_function(np.ndarray[double, ndim=1, mode='c'] x):
	return expanded_scaffer6_func(&x[0], len(x))

cpdef double zakharov_function(np.ndarray[double, ndim=1, mode='c'] x):
	return zakharov_func(&x[0], len(x))

cpdef double discus_function(np.ndarray[double, ndim=1, mode='c'] x):
	return discus_func(&x[0], len(x))

cpdef double grie_rosen_function(np.ndarray[double, ndim=1, mode='c'] x):
	return grie_rosen_func(&x[0], len(x))

cpdef double alpine1_function(np.ndarray[double, ndim=1, mode='c'] x):
	return alpine1_func(&x[0], len(x))

cpdef double alpine2_function(np.ndarray[double, ndim=1, mode='c'] x):
	return alpine2_func(&x[0], len(x))

cpdef double chungreynolds_function(np.ndarray[double, ndim=1, mode='c'] x):
	return chungreynolds_func(&x[0], len(x))

cpdef double cosinemixture_function(np.ndarray[double, ndim=1, mode='c'] x):
	return cosinemixture_func(&x[0], len(x))

cpdef double csendes_function(np.ndarray[double, ndim=1, mode='c'] x):
	return csendes_func(&x[0], len(x))

cpdef double expanded_griewank_plus_rosenbrock_function(np.ndarray[double, ndim=1, mode='c'] x):
	return expanded_griewank_plus_rosenbrock_func(&x[0], len(x))

cpdef double infinity_function(np.ndarray[double, ndim=1, mode='c'] x):
	return infinity_func(&x[0], len(x))

cpdef double michalewichz_function(np.ndarray[double, ndim=1, mode='c'] x, double m):
	return michalewichz_func(&x[0], len(x), m)

cpdef double perm_function(np.ndarray[double, ndim=1, mode='c'] x, double m):
	return perm_func(&x[0], len(x), m)

cpdef double pinter_function(np.ndarray[double, ndim=1, mode='c'] x):
	return pinter_func(&x[0], len(x))

cpdef double powell_function(np.ndarray[double, ndim=1, mode='c'] x):
	return powell_func(&x[0], len(x))

cpdef double qing_function(np.ndarray[double, ndim=1, mode='c'] x):
	return qing_func(&x[0], len(x))

cpdef double quintic_function(np.ndarray[double, ndim=1, mode='c'] x):
	return quintic_func(&x[0], len(x))

cpdef double ridge_function(np.ndarray[double, ndim=1, mode='c'] x):
	return ridge_func(&x[0], len(x))

cpdef double salomon_function(np.ndarray[double, ndim=1, mode='c'] x):
	return salomon_func(&x[0], len(x))

cpdef double sphere2_function(np.ndarray[double, ndim=1, mode='c'] x):
	return sphere2_func(&x[0], len(x))

cpdef double sphere3_function(np.ndarray[double, ndim=1, mode='c'] x):
	return sphere3_func(&x[0], len(x))

cpdef double schwefel_function(np.ndarray[double, ndim=1, mode='c'] x):
	return schwefel_func(&x[0], len(x))

cpdef double schwefel1221_function(np.ndarray[double, ndim=1, mode='c'] x):
	return schwefel1221_func(&x[0], len(x))

cpdef double schwefel1222_function(np.ndarray[double, ndim=1, mode='c'] x):
	return schwefel1222_func(&x[0], len(x))

cpdef double step_function(np.ndarray[double, ndim=1, mode='c'] x):
	return step_func(&x[0], len(x))

cpdef double step2_function(np.ndarray[double, ndim=1, mode='c'] x):
	return step2_func(&x[0], len(x))

cpdef double step3_function(np.ndarray[double, ndim=1, mode='c'] x):
	return step3_func(&x[0], len(x))

cpdef double stepint_function(np.ndarray[double, ndim=1, mode='c'] x):
	return stepint_func(&x[0], len(x))

cpdef double styblinskitang_function(np.ndarray[double, ndim=1, mode='c'] x):
	return styblinskitang_func(&x[0], len(x))

cpdef double trid_function(np.ndarray[double, ndim=1, mode='c'] x):
	return trid_func(&x[0], len(x))

cpdef double whitley_function(np.ndarray[double, ndim=1, mode='c'] x):
	return whitley_func(&x[0], len(x))

cpdef double schumer_steiglitz_function(np.ndarray[double, ndim=1, mode='c'] x):
	return schumer_steiglitz_func(&x[0], len(x))

cpdef double schaffern2_function(np.ndarray[double, ndim=1, mode='c'] x):
	if len(x) < 2: return np.inf
	return schaffern2_func(&x[0])

cpdef double schaffern4_function(np.ndarray[double, ndim=1, mode='c'] x):
	if len(x) < 2: return np.inf
	return schaffern4_func(&x[0])

cpdef double easom_function(np.ndarray[double, ndim=1, mode='c'] x, double a, double b, double c):
	return easom_func(&x[0], len(x), a, b, c)

cpdef double deflected_corrugated_spring_function(np.ndarray[double, ndim=1, mode='c'] x, double alpha, double K):
	return deflected_corrugated_spring_func(&x[0], len(x), alpha, K)

cpdef double needle_eye_function(np.ndarray[double, ndim=1, mode='c'] x, double eye):
	return needle_eye_func(&x[0], len(x), eye)

cpdef double bohachevsky_function(np.ndarray[double, ndim=1, mode='c'] x):
	return bohachevsky_func(&x[0], len(x))

cpdef double deb01_function(np.ndarray[double, ndim=1, mode='c'] x):
	return deb01_func(&x[0], len(x))

cpdef double deb02_function(np.ndarray[double, ndim=1, mode='c'] x):
	return deb02_func(&x[0], len(x))

cpdef double exponential_function(np.ndarray[double, ndim=1, mode='c'] x):
	return exponential_func(&x[0], len(x))

cpdef double xin_she_yang_01_function(np.ndarray[double, ndim=1, mode='c'] x, np.ndarray[double, ndim=1, mode='c'] epsilon):
	if len(x) > len(epsilon): return np.inf
	return xin_she_yang_01_func(&x[0], len(x), &epsilon[0])

cpdef double xin_she_yang_02_function(np.ndarray[double, ndim=1, mode='c'] x):
	return xin_she_yang_02_func(&x[0], len(x))

cpdef double xin_she_yang_03_function(np.ndarray[double, ndim=1, mode='c'] x, double beta, double m):
	return xin_she_yang_03_func(&x[0], len(x), beta, m)

cpdef double xin_she_yang_04_function(np.ndarray[double, ndim=1, mode='c'] x):
	return xin_she_yang_04_func(&x[0], len(x))

cpdef double yaoliu_09_function(np.ndarray[double, ndim=1, mode='c'] x):
	return yaoliu_09_func(&x[0], len(x))

cpdef double clustering_function(int a, np.ndarray[double, ndim=1, mode='c'] o, int n, np.ndarray[double, ndim=1, mode='c'] z, int k, np.ndarray[double, ndim=1, mode='c'] w):
	return clustering_func(a, &o[0], n, &z[0], k, &w[0])

cpdef double clustering_min_function(int a, np.ndarray[double, ndim=1, mode='c'] o, int n, np.ndarray[double, ndim=1, mode='c'] z, int k, np.ndarray[double, ndim=1, mode='c'] w):
	return clustering_min_func(a, &o[0], n, &z[0], k, &w[0])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
