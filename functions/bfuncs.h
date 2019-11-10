#ifndef BFUNCS_H_   /* Include guard */
#define BFUNCS_H_

#include <stdbool.h>

#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

double ackley_func (double*, int);
double sphere_func (double*, int);
double ellips_func (double*, int);
double bent_cigar_func (double*, int);
double dixon_price_func (double*, int);
double griewank_func (double*, int);
double happycat_func (double*, int);
double hgbat_func (double*, int);
double katsuura_func (double*, int);
double levy_func (double*, int);
double rastrigin_func (double*, int);
double rosenbrock_func (double*, int);
double weierstrass_func (double*, int);
double modified_schwefel_func (double*, int);
double expanded_scaffer6_func (double*, int);
double zakharov_func (double*, int);
double discus_func (double*, int);
double grie_rosen_func (double*, int);
double alpine1_func (double*, int);
double alpine2_func (double*, int);
double chungreynolds_func (double*, int);
double cosinemixture_func (double*, int);
double csendes_func (double*, int);
double expanded_griewank_plus_rosenbrock_func (double*, int);
double infinity_func (double*, int);
double michalewichz_func (double*, int, double);
double perm_func (double*, int, double);
double pinter_func (double*, int);
double powell_func (double*, int);
double qing_func (double*, int);
double quintic_func (double*, int);
double ridge_func (double*, int);
double salomon_func (double*, int);
double sphere2_func (double*, int);
double sphere3_func (double*, int);
double schwefel_func (double*, int);
double schwefel1222_func (double*, int);
double schwefel1221_func (double*, int);
double step_func (double*, int);
double step2_func (double*, int);
double step3_func (double*, int);
double stepint_func (double*, int);
double styblinskitang_func (double*, int);
double trid_func (double*, int);
double whitley_func (double*, int);
double schumer_steiglitz_func (double*, int);
double deflected_corrugated_spring_func (double*, int, double, double);
double needle_eye_func (double*, int, double);
double bohachevsky_func (double*, int);
double deb01_func (double*, int);
double deb02_func (double*, int);
double exponential_func (double*, int);
double xin_she_yang_01_func (double*, int, double*);
double xin_she_yang_02_func (double*, int);
double xin_she_yang_03_func (double*, int, double, double);
double xin_she_yang_04_func (double*, int);
double yaoliu_09_func (double*, int);
// Only two variables in use 
double schaffern2_func (double*);
double schaffern4_func (double*);

double easom_func (double*, int, double, double, double);

double Tchebyshev(double*, int);
double Hilbert(double*, int);
double Lennard_Jones(double*, int);

double molecular_ab_initio_model(double*, int);
double molecular_hp_qibic_model(int*, int);
double lowest_auto_correlation_bit_sequence(bool*, int);

double clustering_func (int, double*, int, double*, int, double*);
double clustering_min_func (int, double*, int, double*, int, double*);

#endif // BFUNCS_H_

// vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
