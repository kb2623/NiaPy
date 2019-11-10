#include <math.h>
#include "bfuncs.h"

/** Sphere
 *
 * @brief sphere_func
 * @param x Input array.
 * @param nx Number of elements is array.
 * @return Fitness/function value of input array.
 */
double sphere_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i];
	return f;
}

/** Ackley's
 *
 * @brief ackley_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double ackley_func (double* x, int nx) {
	int i;
	double sum1 = 0.0, sum2 = 0.0;
	for (i = 0; i < nx; i++) {
		sum1 += x[i] * x[i];
		sum2 += cos(2.0 * PI * x[i]);
	}
	sum1 = -0.2 * sqrt(sum1 / nx);
	sum2 /= nx;
	return E - 20.0 * exp(sum1) - exp(sum2) + 20.0;
}

/** Ellipsoidal
 *
 * @brief ellips_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double ellips_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(10.0, 6.0 * i / (nx - 1)) * x[i] * x[i];
	return f;
}

/** Bent_Cigar
 *
 * @brief bent_cigar_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double bent_cigar_func (double* x, int nx) {
	int i;
	double f = x[0] * x[0];
	for (i = 1; i < nx; i++) f += pow(10.0, 6.0) * x[i] * x[i];
	return f;
}

/** Dixon and Price
 *
 * @brief dixon_price_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double dixon_price_func (double* x, int nx) {
	int i, j;
	double term1 = pow((x[0] - 1), 2), sum = 0;
	for (i = 1; i < nx; i++) {
		double xi = x[i], xold = x[i-1];
		double newv = i * pow(2 * pow(xi, 2) - xold, 2);
		sum = sum + newv;
	}
	return term1 + sum;
}

/** Griewank's
 *
 * @brief griewank_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double griewank_func (double* x, int nx) {
	int i;
	double s = 0.0, p = 1.0;
	for (i = 0; i < nx; i++) {
		s += x[i] * x[i];
		p *= cos(x[i] / sqrt(1.0 + i));
	}
	return 1.0 + s / 4000.0 - p;
}

/** HappyCat, provdided by Hans-Georg Beyer (HGB)
 * original global optimum: [-1,-1,...,-1]
 *
 * @brief happycat_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double happycat_func (double* x, int nx) {
	int i;
	double alpha = 1.0 / 8.0, r2 = 0.0, sum_z = 0.0;
	for (i = 0; i < nx; i++) {
		r2 += x[i] * x[i];
		sum_z += x[i];
	}
	return pow(fabs(r2 - nx), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5;
}

/** HGBat, provdided by Hans-Georg Beyer (HGB)
 * original global optimum: [-1,-1,...,-1]
 *
 * @brief hgbat_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double hgbat_func (double* x, int nx) {
	int i;
	double alpha = 1.0 / 4.0, r2 = 0.0, sum_z = 0.0;
	for (i = 0; i < nx; i++) {
		r2 += x[i] * x[i];
		sum_z += x[i];
	}
	return pow(fabs(pow(r2, 2.0) - pow(sum_z, 2.0)), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5;
}

/** Katsuura
 *
 * @brief katsuura_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double katsuura_func (double* x, int nx) {
	int i,j;
	double temp, tmp1, tmp2, tmp3;
	double f = 1.0;
	tmp3=pow(1.0 * nx, 1.2);
	for (i = 1; i < nx; i++) {
		temp=0.0;
		for (j = 1; j <= 32; j++) {
			tmp1 = pow(2.0, j);
			tmp2 = tmp1 * x[i];
			temp += fabs(tmp2 - floor(tmp2 + 0.5)) / tmp1;
		}
		f *= pow(1.0 + (i + 1) * temp, 10.0 / tmp3);
	}
	tmp1=10.0 / nx / nx;
	return f * tmp1 - tmp1;
}

/** Levy
 *
 * @brief levy_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double levy_func (double* x, int nx) {
	int i;
	double f = 0.0;
	double *w;
	w=(double *)malloc(sizeof(double)  *  nx);
	double sum1= 0.0;
	for (i = 0; i < nx; i++) w[i] = 1.0 + (x[i] - 1.0) / 4.0;
	double term1 = pow((sin(PI * w[0])), 2);
	double term3 = pow((w[nx - 1] - 1), 2) * (1 + pow((sin(2 * PI * w[nx - 1])), 2));
	double sum = 0.0;
	for (i = 0; i < nx-1; i++) {
		double wi = w[i];
		double newv = pow((wi - 1), 2) * (1 + 10 * pow((sin(PI * wi + 1)), 2));
		sum = sum + newv;
	}
	free(w);   // ADD THIS LINE to free memory! Thanks for Dr. Janez
	return term1 + sum + term3;
}

/** Rastrigin's
 *
 * @brief rastrigin_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double rastrigin_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += (x[i] * x[i] - 10.0 * cos(2.0 * PI * x[i]) + 10.0);
	return f;
}

/** Rosenbrock's
 *
 * @brief rosenbrock_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double rosenbrock_func (double* x, int nx) {
	int i;
	double tmp1,tmp2, f = 0.0;
	for (i = 0; i < nx - 1; i++) {
		tmp1 = x[i] * x[i] - x[i+1];
		tmp2 = x[i] - 1.0;
		f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
	}
	return f;
}

/** Weierstrass's
 *
 * @brief weierstrass_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double weierstrass_func (double* x, int nx) {
	int i,j,k_max;
	double sum,sum2, a, b;
	a = 0.5;
	b = 3.0;
	k_max = 20;
	double f = 0.0;
	for (i = 0; i < nx; i++) {
		sum = 0.0, sum2 = 0.0;
		for (j = 0; j <= k_max; j++) {
			sum += pow(a, j) * cos(2.0 * PI * pow(b, j) * (x[i] + 0.5));
			sum2 += pow(a, j) * cos(2.0 * PI * pow(b, j) * 0.5);
		}
		f += sum;
	}
	return f - nx * sum2;
}

/** Lennard Jones potencial.
 *
 * Find the atomic configuration with minimum energy
 *
 * valid for any dimension, D=3*k, k=2,3,4,...,25.   k is the number of atoms in 3-D space
 * constraints: unconstrained
 * type: multi-modal with one global minimum; non-separable
 * initial upper bound = 4, initial lower bound = -4
 * value-to-reach = minima[k-2]+.0001
 * f(x*) = minima[k-2]; see array of minima below; additional minima available at the
 * Cambridge cluster database: http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
 *
 * @brief Lennard_Jones
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double Lennard_Jones(double* x, int nx) {
	static double minima[] = { -1.,-3.,-6.,-9.103852,-12.712062,-16.505384,-19.821489,-24.113360, -28.422532,-32.765970,-37.967600,-44.326801,-47.845157,-52.322627,-56.815742,-61.317995, -66.530949,-72.659782,-77.1777043,-81.684571,-86.809782,-02.844472,-97.348815,-102.372663 };
	int i, j, k, a, b;
	long double xd, yd, zd, ed, ud, sum = 0;
	k = nx / 3;
	for (i = 0; i < k - 1; i++) {
		for (j = i + 1; j < k; j++) {
			a = 3 * i, b = 3 * j;
			xd = x[a] - x[b], yd = x[a + 1] - x[b + 1], zd = x[a + 2] - x[b + 2];
			ed = xd * xd + yd * yd + zd * zd;
			ud = ed * ed * ed;
			if (ud > 1.0e-10) sum += (1.0 / ud - 2.0) / ud;
			else sum += 1.0e20;
		}
	}
	return sum;
}

/** Weierstrass's
 *
 * @brief weierstrass_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double molecular_ab_initio_model(double* x, int nx) {
	// TODO
	return 0;
}

/** Weierstrass's
 *
 * @brief weierstrass_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double molecular_hp_qibic_model(int* x, int nx) {
	// TODO
	return 0;
}

/** Weierstrass's
 *
 * @brief weierstrass_func
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double lowest_auto_correlation_bit_sequence(bool* x, int nx) {
	// TODO
	return 0;
}

/**find the inverse of the (ill-conditioned) Hilbert matrix
 *
 * valid for any dimension, n=k*k, k=2,3,4,...
 * constraints: unconstrained
 * type: multi-modal with one global minimum; non-separable
 * initial upper bound = 2^n, initial lower bound = -2^n
 * value-to-reach = f(x*)+1.0e-8
 * f(x*) = 0.0; x*={{9,-36,30},{-36,192,-180},{30,-180,180}} (n=9)
 * x*={{16,-120,240,-140},{-120,1200,-2700,1680},{240,-2700,6480,4200},{-140,1680,-4200,2800}} (n=16)
 * 
 * @brief Hilbert
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double Hilbert(double* x, int nx) {
	int i, j, k, b;
	long double sum = 0;
	b = (int) sqrt((double) nx);
	long double hilbert[b][b], y[b][b];			// Increase matrix size if D > 100
	for (i = 0; i < b; i++) {
		for (j = 0; j < b; j++) hilbert[i][j] = 1. / (double)(i + j + 1);		// Create a static Hilbert matrix
	}
	for (j = 0; j < b; j++) {
		for (k = 0; k < b; k++) {
			y[j][k] = 0;
			for (i = 0; i < b; i++) y[j][k] += hilbert[j][i] * x[k + b * i];		// Compute matrix product H*x
		}
	}
	for (i = 0; i < b; i++) {
		for (j = 0; j < b; j++) {
			if (i == j) sum += fabs(y[i][j] - 1);				// Sum absolute value of deviations
			else sum += fabs(y[i][j]);
		}
	}
	return sum;
}

/** Storn's Tchebychev - a 2nd ICEO function - generalized version
 * 
 * Valid for any D>2
 * constraints: unconstrained
 * type: multi-modal with one global minimum; non-separable
 * initial upper bound = 2^D, initial lower bound = -D^n
 * value-to-reach = f(x*)+1.0e-8
 * f(x*)=0.0; x*=(128,0,-256,0,160,0,-32,0,1) (n=9)
 * x*=(32768,0,-131072,0,212992,0,-180224,0,84480,0,-21504,0,2688,0,-128,0,1) (n=17)
 * 
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array. 
 */
double Tchebyshev(double* x, int nx) {
	int i, j;
	static int sample;
	long double a = 1., b = 1.2, px, y = -1, sum = 0;
	static long double dx, dy;
	for (j = 0; j < nx - 2; j++) {
		dx = 2.4 * b - a;
		a = b, b = dx;
	}
	sample = 32 * nx;
	dy = 2. / (long double)sample;
	for (i = 0; i <= sample; i++) {
		px = x[0];
		for (j = 1; j < nx; j++) px = y * px + x[j];
		if (px < -1 || px > 1) sum += (1. - fabs(px)) * (1. - fabs(px));
		y += dy;
	}
	for (i = -1; i <= 1; i += 2) {
		px = x[0];
		for (j = 1; j < nx; j++) px = 1.2 * px + x[j];
		if (px < dx) sum += px * px;
	}
	return sum;
}

/* Modified Schwefel's
 *
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double modified_schwefel_func (double* x, int nx) {
	int i;
	double tmp, z;
	double f = 0.0;
	for (i = 0; i < nx; i++) {
		z = x[i] + 4.209687462275036e+002;
		if (z > 500) {
			f -= (500.0 - fmod(z, 500)) * sin(pow(500.0 - fmod(z, 500), 0.5));
			tmp = (z - 500.0) / 100;
			f += tmp * tmp / nx;
		} else if (z < -500) {
			f -= (-500.0 + fmod(fabs(z), 500)) * sin(pow(500.0 - fmod(fabs(z), 500), 0.5));
			tmp = (z + 500.0) / 100;
			f += tmp * tmp / nx;
		} else {
			f -= z * sin(pow(fabs(z), 0.5));
		}
	}
	return f + 4.189828872724338e+002 * nx;
}

/* Expanded Scaffer's F6
 *
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double expanded_scaffer6_func (double* x, int nx) {
	int i;
	double temp1, temp2;
	double f = 0.0;
	for (i = 0; i < nx-1; i++) {
		temp1 = sin(sqrt(x[i] * x[i] + x[i+1] * x[i+1]));
		temp1 = temp1 * temp1;
		temp2 = 1.0 + 0.001 * (x[i] * x[i] + x[i+1] * x[i+1]);
		f += 0.5 + (temp1 - 0.5) / (temp2 * temp2);
	}
	temp1 = sin(sqrt(x[nx - 1] * x[nx - 1] + x[0] * x[0]));
	temp1 = temp1 * temp1;
	temp2 = 1.0 + 0.001 * (x[nx - 1] * x[nx - 1] + x[0] * x[0]);
	return f + 0.5 + (temp1 - 0.5) / (temp2 * temp2);
}

/* zakharov
 *
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double zakharov_func (double* x, int nx)  {
	int i;
	double sum1 = 0.0, sum2 = 0.0;
	for (i = 0; i < nx; i++) {
		sum1 += pow(x[i], 2);
		sum2 += 0.5 * (i + 1) * x[i];
	}
	return sum1 + pow(sum2, 2) + pow(sum2, 4);
}

/* Discus 
 *
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double discus_func (double* x, int nx) {
	int i;
	double f = pow(10.0, 6.0) * x[0]* x[0];
	for (i = 1; i < nx; i++) f += x[i] * x[i];
	return f;
}

/* Griewank-Rosenbrock
 *
 * @brief Tchebyshev
 * @param x Input array.
 * @param nx Number of elements in array.
 * @return Fitness/function value of input array.
 */
double grie_rosen_func (double* x, int nx) {
	int i;
	double temp, tmp1, tmp2;
	double f = 0.0;
	for (i = 0; i < nx - 1; i++) {
		tmp1 = x[i] * x[i] - x[i + 1];
		tmp2 = x[i] - 1.0;
		temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
		f += (temp * temp) / 4000.0 - cos(temp) + 1.0;
	}
	tmp1 = x[nx - 1] * x[nx - 1] - x[0];
	tmp2 = x[nx - 1] - 1.0;
	temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
	return f + (temp * temp) / 4000.0 - cos(temp) + 1.0;
}

double alpine1_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += fabs(sin(x[i]) * 0.1 * x[i]);
	return f;
}

double alpine2_func (double* x, int nx) {
	int i;
	double f = 1.0;
	for (i = 0; i < nx; i++) f *= sqrt(x[i]) * sin(x[i]);
	return f;
}

double chungreynolds_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i];
	return f * f;
}

double cosinemixture_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 0.0;
	for (i = 0; i < nx; i++) f1 += cos(5 * PI * x[i]), f2 += x[i] * x[i];
	return -0.1 * f1 - f2;
}

double csendes_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) {
		if (x[i] != 0) f += pow(x[i], 6) * (2 * sin(1 / x[i]));
	}
	return f;
}

double expanded_griewank_plus_rosenbrock_h (double x) {
	return x * x / 4000 - cos(x / sqrt(1)) + 1;
}

double expanded_griewank_plus_rosenbrock_g (double x, double y) {
	return 100 * pow(x * x - y * y, 2) + pow(x - 1, 2);
}

double expanded_griewank_plus_rosenbrock_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 1; i < nx; i++) f += expanded_griewank_plus_rosenbrock_h(expanded_griewank_plus_rosenbrock_g(x[i - 1], x[i]));
	return f;
}

double infinity_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(x[i], 6) * (sin(1 / x[i]) + 2);
	return f;
}

double michalewichz_func (double* x, int nx, double m) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += sin(x[i]) * pow(sin((i + 1) * x[i] * x[i] / PI), 2 * m);
	return -f;
}

double perm_func (double* x, int nx, double beta) {
	int i, j;
	double f = 0.0, ff;
	for (i = 0; i < nx; i++) {
		ff = 0.0;
		for (j = 0; j < nx; j++) ff += (j + 1 + beta) * (pow(x[j], i + 1) - 1 / pow(j + 1, i + 1));
		f += ff * ff;
	}
	return f;
}

double pinter_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 0.0, f3 = 0.0;
	for (i = 0; i < nx; i++) {
		double sub, add;
		if (i == 0) sub = x[nx - 1], add = x[i + 1];
		else if (i == nx - 1) sub = x[nx - 1], add = x[0];
		else sub = x[i - 1], add = x[i + 1];
		double a = sub * sin(x[i]) + sin(add);
		double b = sub * sub - 2 * x[i] + 3 * add - cos(x[i]) + 1;
		f1 += (i + 1) * x[i] * x[i];
		f2 += 20 * (i + 1) * pow(sin(a), 2);
		f3 += (i + 1) * log10(1 + (i + 1) * b * b);
	}
	return f1 + f2 + f3;
}

double powell_func (double* x, int nx) {
	int i, len = nx / 4;
	double f = 0.0;
	for (i = 1; i <= len; i++) f += pow(x[4 * i - 4] + 10 * x[4 * i - 3], 2) + 5 * pow(x[4 * i - 2] - x[4 * i - 1], 2) + pow(x[4 * i - 3] - 2 * x[4 * i - 2], 4) + 10 * pow(x[4 * i - 4] - x[4 * i - 1], 4);
	return f;
}

double qing_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(pow(x[i], 2) - i, 2);
	return f;
}

double quintic_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += fabs(pow(x[i], 5) - 3 * pow(x[i], 4) + 4 * pow(x[i], 3) + 2 * pow(x[i], 2) - 10 * x[i] - 4);
	return f;
}

double ridge_func (double* x, int nx) {
	int i, j;
	double f = 0.0, ff;
	for (i = 0; i < nx; i++) {
		ff = 0.0;
		for (j = 0; j < i + 1; j++) ff += x[j];
		f += ff * ff;
	}
	return f;
}

double salomon_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i];
	return 1 - cos(2 * PI * sqrt(f)) + 0.1 * f;
}

double sphere2_func (double* x, int nx) {
	int i, j;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(x[i], i + 1);
	return f;
}

double sphere3_func (double* x, int nx) {
	int i, j;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i];
	return nx * f;
}

double schwefel_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * sin(sqrt(x[i]));
	return 418.9829 * nx - f;
}

double schwefel1222_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 1.0;
	for (i = 0; i < nx; i++) f1 += fabs(x[i]), f2 *= fabs(x[i]);
	return f1 + f2;
}

double schwefel1221_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f = (f < fabs(x[i])) ? x[i] : f;
	return f;
}

double step_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += floor(fabs(x[i]));
	return f;
}

double step2_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(floor(x[i] + 0.5), 2);
	return f;
}

double step3_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += floor(pow(x[i], 2));
	return f;
}

double stepint_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += floor(x[i]);
	return 25 + f;
}

double styblinskitang_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(x[i], 4) - 16 * x[i] * x[i] + 5.0 * x[i];
	return f * 0.5;
}

double trid_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 0.0;
	for (i = 0; i < nx; i++) {
		f1 += pow(x[i] - 1, 2);
		if (i > 0) f2 += x[i] * x[i - 1];
	}
	return f1 - f2;
}

double whitley_func (double* x, int nx) {
	int i, j;
	double f = 0.0;
	for (i = 0; i < nx; i++) {
		for (j = 0; j < nx; j++) {
			double tmp = 100 * pow(x[i] * x[i] - x[j], 2) + pow(1 - x[j], 2);
			f += (tmp * tmp) / 4000 - cos(tmp) + 1;
		}
	}
	return f;
}

double schaffern2_func (double* x) {
	return 0.5 + (pow(sin(x[0] * x[0] - x[1] * x[1]), 2) - 0.5) / pow(1 + 0.001 * (x[0] * x[0] + x[1] * x[1]), 2);
}

double schaffern4_func (double *x) {
	return 0.5 + (pow(cos(sin(x[0] * x[0] - x[1] * x[1])), 2) - 0.5) / pow(1 + 0.001 * (x[0] * x[0] + x[1] * x[1]), 2);
}

double schumer_steiglitz_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i] * x[i] * x[i];
	return f;
}

double easom_func (double* x, int nx, double a, double b, double c) {
	int i;
	double f1 = 0.0, f2 = 0.0;
	for (i = 0; i < nx; i++) f1 += x[i] * x[i], f2 += cos(c * x[i]);
	return a - (a / exp(b * sqrt(f1 / nx))) + E - exp(f2 / nx);
}

double deflected_corrugated_spring_func (double* x, int nx, double alpha, double K) {
	int i;
	double f = 0.0, ff = 0.0;
	for (i = 0; i < nx; i++) ff += pow(x[i] - alpha, 2);
	for (i = 0; i < nx; i++) f += pow(x[i] - alpha, 2) - cos(K * sqrt(ff));
	return 0.1 * f;
}

double needle_eye_func (double* x, int nx, double eye) {
	int i;
	double f = 0.0, ff = 0.0;
	for (i = 0; i < nx; i++) ff += 100 + fabs(x[i]);
	for (i = 0; i < nx; i++) {
		if (fabs(x[i]) == eye) f += 0;
		else if (fabs(x[i]) > eye) f += ff;
		else f += 1;
	}
	return f;
}

double bohachevsky_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx - 1; i++) f += x[i] * x[i] + 2 * (x[i + 1] * x[i + 1]) - 0.3 * cos(3 * PI * x[i]) - 0.4 * cos(4 * PI * x[i + 1]) + 0.7;
	return f;
}

double deb01_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(sin(5 * PI * x[i]), 6);
	return - f / nx;
}

double deb02_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += pow(sin(5 * PI * (pow(sqrt(sqrt(x[i])), 3) - 0.05)), 6);
	return - f / nx;
}

double exponential_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i];
	return -exp(-0.5 * f);
}

double xin_she_yang_01_func (double* x, int nx, double* epsilon) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += epsilon[i] * pow(fabs(x[i]), i + 1);
	return f;
}

double xin_she_yang_02_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 0.0;
	for (i = 0; i < nx; i++) f1 += fabs(x[i]), f2 += sin(x[i] * x[i]);
	return f1 * exp(-f2);
}

double xin_she_yang_03_func (double* x, int nx, double beta, double m) {
	int i;
	double f1 = 0.0, f2 = 0.0, f3 = 1.0;
	for (i = 0; i < nx; i++) f1 += pow(x[i] / beta, 2 * m), f2 += x[i] * x[i], f3 *= pow(cos(x[i]), 2);
	return exp(-f1) - 2 * exp(-f2) * f3;
}

double xin_she_yang_04_func (double* x, int nx) {
	int i;
	double f1 = 0.0, f2 = 0.0, f3 = 0.0;
	for (i = 0; i < nx; i++) f1 += x[i] * x[i], f2 += pow(sin(x[i]), 2), f3 += pow(sin(sqrt(fabs(x[i]))), 2);
	return f2 - exp(-f1) * exp(-f3);
}

double yaoliu_09_func (double* x, int nx) {
	int i;
	double f = 0.0;
	for (i = 0; i < nx; i++) f += x[i] * x[i] - 10 * cos(2 * PI * x[i]) + 10;
	return f;
}

/**Function for optimizing clusters.
 *
 * @brief Clustering optimization
 * @param a Number atributtes
 * @param O Dataset
 * @param n Number of rows in dataset
 * @param Z Clusters definition
 * @param k Number of clusters
 * @return Sum of euclidian distances for all clusters on dataset
 */
double clustering_func (int a, double* O, int n, double* Z, int k, double* W) {
	double s = 0, ss = 0;
	int i, j, l;
	for (i = 0; i < n; i++) {
		for (j = 0; j < k; j++) {
			ss = 0;
			for (l = 0; l < a; l++) ss += pow(O[a * i + l] - Z[a * j + l], 2);
			s += W[i * k + j] * sqrt(ss);
		}
	}
	return s;
}

/**Function two for optimizing clusters.
 *
 * @brief Clustering optimization
 * @param a Number atributtes
 * @param O Dataset
 * @param n Number of rows in dataset
 * @param Z Clusters definition
 * @param k Number of clusters
 * @return Sum of euclidian distances for all min clusters on dataset
 */
double clustering_min_func (int a, double* O, int n, double* Z, int k, double* W) {
	double s = 0, ss = 0, tmp = 0, min = -1;
	int i, j, l;
	for (i = 0; i < n; i++) {
		for (j = 0; j < k; j++) {
			ss = 0;
			for (l = 0; l < a; l++) ss += pow(O[a * i + l] - Z[a * j + l], 2);
			tmp = W[i * k + j] * sqrt(ss);
			if (min == -1 || tmp < min) min = tmp;
		}
		s += min;
	}
	return s;
}

// vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
