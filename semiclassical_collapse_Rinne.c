///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
This code was written by Benjamin Berczi as part of the PhD project titled "Simulating Semiclassical Black Holes" from the University of Nottingham.

It is a self-contained C file that simulates a massless quantum scalar field coupled to Einstein gravity in the ADM formulation.

Details may be found in Benjamin Berczi's publications and PhD thesis.

*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initialising the libraries and constants for the code */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <complex.h>                                                              

/* CONSANTS */
#define         PI                                 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define         M_P                                1.0                                                                                  //sqrt(8*PI)
#define         c                                  1.0                                                                                  //speed of light

/* GRID PARAMETERS */
#define         lattice_size                       250
#define         r_max                              5.6
#define         r_eff                              r_max/1.0
#define         dx                                 1.0/(lattice_size-0.5)

/* SCALAR FIELD PARAMETERS */
#define         amplitude                          0.0 
#define         mass                               0.0
#define         initial_radius                     0.0                                                                                   // initial radius of the gaussian scalar field
#define         initial_width                      1.0                                                                                   // initial width of the gaussian scalar field

/* QUANTUM OR CLASSICAL SIMULATION */
#define         hbar                               0                                                                                     // set to 1 for quantum, 0 for classical. This just sets the backreaction, and is in set_bi_linears.c, the quantum modes are still evolved
#define         coherent_state_switch              1                                                                                     // set to 0 to just have the mode functions

/* QUANTUM GHOST FIELD PARAMETERS */
#define         number_of_q_fields                 6                                                                                     // number of quantum fields, 1 real, 5 ghosts for regularisation
#define         muSq                               0.0                                                                                   // mass of scalar field
#define         mSqGhost                           1.0                                                                                   // base mass of the Pauli-Villars regulator fields
double          massSq[number_of_q_fields] = { muSq, mSqGhost, 3.0 * mSqGhost, mSqGhost, 3.0 * mSqGhost, 4.0 * mSqGhost };               // masses of the ghost fields
double          ghost_or_physical[6] = { 1 , -1 , 1 , -1 , 1, -1 };                                                                      // distinguishing between the real and ghost fields

/* QUANTUM MODE PARAMETERS */
#define         k_min                              1.0*PI/15.0                                                                           // minimum value of k, also =dk
#define         dk                                 k_min            
#define         number_of_k_modes                  1                                                                                     // number of k modes
#define         number_of_l_modes                  1                                                                                     // number of l modes
#define         k_start                            0
#define         l_start                            0                                                                                     // the range of l is l_start, l_start+l_step, l_start+2l_step...
#define         l_step                             1

/* SIMULATION PARAMETERS */
#define         evolve_time_int                    200
#define         nu                                 1.0
#define         lambda                             0.5
#define         reps                               1
#define         divid                              1
#define         epsilon                            0.15                                                                                 // epsilon is the constant in the damping term, it's max value is 0.5
#define         ep2                                50.0
#define         r_damp                             100000000000000000.0

/* NUMERICAL TIME ITERATION PARAMETERS */
#define         nu_legendre                        5
#define         number_of_RK_implicit_iterations   10


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions that saves data*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_rpoints(double* field) {
    FILE* finout;
    finout = fopen("rpoints250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_chi(double* field) {
    FILE* finout;
    finout = fopen("chi250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_pi(double* field) {
    FILE* finout;
    finout = fopen("pi250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_beta(double* field) {
    FILE* finout;
    finout = fopen("beta250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_Khat(double* field) {

    FILE* finout;
    finout = fopen("Khat250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_psi(double* field) {

    FILE* finout;
    finout = fopen("psi250.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_psi_prime(double* field) {

    FILE* finout;
    finout = fopen("psi_prime.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_psi_constr(double field[lattice_size]) {

    FILE* finout;
    finout = fopen("psi_constr1.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_K_constr(double field[lattice_size]) {

    FILE* finout;
    finout = fopen("K_constr1.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_alphar(double* field) {

    FILE* finout;
    finout = fopen("alpha.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {        
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_alpha_prime(double* field) {

    FILE* finout;
    finout = fopen("alpha_prime1.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_alpha(double* field) {

    FILE* finout;
    finout = fopen("alphat3.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_horizon(double* field) {

    FILE* finout;
    finout = fopen("horizonr.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_time(double* field) {

    FILE* finout;
    finout = fopen("horizont.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_constr_t(double* field) {

    FILE* finout;
    finout = fopen("constr_t.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_field_t(double** field) {

    FILE* finout;
    finout = fopen("ham_mat250.txt", "w");
    for (int n = 0; n < evolve_time_int; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.200f ", field[n][m]);
        }
    }
    fclose(finout);
}
void save_r_mat(double** r) {

    FILE* finout;
    finout = fopen("rmat.txt", "w");
    for (int n = 0; n < evolve_time_int; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.200f ", r[n][m]);
        }
    }
    fclose(finout);
}
void save_t_mat(double** t) {

    FILE* finout;
    finout = fopen("tmat.txt", "w");
    for (int n = 0; n < evolve_time_int; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.200f ", t[n][m]);
        }
    }
    fclose(finout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Creating a structure for the variables */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct classical_fields {
    double* pi;
    double* chi;
};
typedef struct classical_fields Classical_fields;
struct quantum_fields {
    __complex__ double*** pi;
    __complex__ double*** phi;
    __complex__ double*** chi;
};
typedef struct quantum_fields Quantum_fields;
struct metric_fields {
    double* alpha;
    double* beta;
    double* psi;
    double* K_hat;
};
typedef struct metric_fields Metric_Fields;
struct dmetric_fields {
    double alpha;
    double beta;
    double psi;
    double K_hat;
};
typedef struct dmetric_fields dMetric_Fields;
struct dmetric_bound_dot {
    double alpha;
    double beta;
    double psi;
    double Khat;
};
typedef struct dmetric_bound_dot dMetric_Bound_Dot;

struct bi_linears {
    double phi_phi;
    double chi_chi_rr;
    double pi_pi;
    double chi_pi;
    double del_theta_phi_del_theta_phi_over_r_sq;
};
typedef struct bi_linears Bi_Linears;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Making the points for the spatial grid*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void make_points(double xp[lattice_size]) {
    
    for (int i = 0; i < lattice_size; ++i) {
        xp[i] = (i + 0.5) * dx;

    }
}
double r_x(double x) {
    return r_eff * x + (r_max - r_eff) * pow(x, 3.0);
}
double dr_dx(double x) {
    return (r_eff + 3.0 * (r_max - r_eff) * x * x);
}
double d2r_dx2(double x) {
    return (6.0 * (r_max - r_eff) * x);
}
double dx_dr(double x) {
    return 1.0 / (r_eff + 3.0 * (r_max - r_eff) * x * x);
}
void save_points(double x[lattice_size]) {
    FILE* pointsout;
    pointsout = fopen("points1.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(pointsout, "%.20f ", r_x((m + 0.5) * dx));
    }
    fclose(pointsout);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Derivative functions for real fields and complex fields separately, uses 20 neighbouring points*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = dx_dr((m+0.5) * dx) * (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m] + field[m + 1]) / (12.0 * dx);
    }
    else if (m == 1) {
        return der = dx_dr((m+0.5) * dx) * (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 1]) / (12.0 * dx);
    }   
    else if (m == lattice_size - 2){
        return der = dx_dr((m + 0.5) * dx) * (-1.0*field[m - 3] + 6.0 * field[m - 2] - 18.0 * field[m - 1] + 10.0 * field[m] + 3.0 * field[m + 1]) / (12.0 * dx);
    }
    else if (m == lattice_size - 1) {
        return der = dx_dr((m + 0.5) * dx) * (3.0 * field[m - 4] - 16.0 * field[m - 3] + 36.0 * field[m - 2] - 48.0 * field[m - 1] + 25.0 * field[m]) / (12.0 * dx);
    }
    else {
        return der = dx_dr((m+0.5) * dx) * (-field[m + 2] + 8.0*field[m+1] - 8.0*field[m-1] + field[m - 2]) / (12.0 * dx);
    }
}
__complex__ double first_deriv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        return der = dx_dr((m + 0.5) * dx) * (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m] + field[m + 1]) / (12.0 * dx);
    }
    else if (m == 1) {
        return der = dx_dr((m + 0.5) * dx) * (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 1]) / (12.0 * dx);
    }
    else if (m == lattice_size - 2) {
        return der = dx_dr((m + 0.5) * dx) * (-1.0 * field[m - 3] + 6.0 * field[m - 2] - 18.0 * field[m - 1] + 10.0 * field[m] + 3.0 * field[m + 1]) / (12.0 * dx);
    }
    else if (m == lattice_size - 1) {
        return der = dx_dr((m + 0.5) * dx) * (3.0 * field[m - 4] - 16.0 * field[m - 3] + 36.0 * field[m - 2] - 48.0 * field[m - 1] + 25.0 * field[m]) / (12.0 * dx);
    }
    else {
        return der = dx_dr((m + 0.5) * dx) * (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 2]) / (12.0 * dx);
    }
}
double second_deriv(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        der = (-field[m + 1] + 16.0 * field[m] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }
    else if (m == 1) {
        der = (-field[m - 1] + 16.0 * field[m - 1] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }
    else if (m == lattice_size - 2) {
        der = (11.0*field[m + 1] - 20.0 * field[m] + 6.0*field[m - 1] + 4.0*field[m-2]-field[m-3]) / (12.0*dx * dx);
    }
    else if (m == lattice_size - 1) {
        der = (11.0 * field[m - 4] - 56.0 * field[m -3] + 114.0 * field[m - 2] - 104.0 * field[m - 1] + 35.0*field[m]) / (12.0*dx * dx);

    }
    else  {
        der = (-field[m - 2] + 16.0 * field[m - 1] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }

    return dx_dr((m+0.5) * dx) * dx_dr((m+0.5) * dx) * (der -d2r_dx2((m + 0.5) * dx) * first_deriv(m, field));
}
__complex__ double second_deriv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        der = (-field[m + 1] + 16.0 * field[m] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }
    else if (m == 1) {
        der = (-field[m - 1] + 16.0 * field[m - 1] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }
    else if (m == lattice_size - 2) {
        der = (11.0 * field[m + 1] - 20.0 * field[m] + 6.0 * field[m - 1] + 4.0 * field[m - 2] - field[m - 3]) / (12.0 * dx * dx);
    }
    else if (m == lattice_size - 1) {
        der = (11.0 * field[m - 4] - 56.0 * field[m - 3] + 114.0 * field[m - 2] - 104.0 * field[m - 1] + 35.0 * field[m]) / (12.0 * dx * dx);

    }
    else {
        der = (-field[m - 2] + 16.0 * field[m - 1] - 30.0 * field[m] + 16.0 * field[m + 1] - field[m + 2]) / (12.0 * dx * dx);
    }

    return dx_dr((m + 0.5) * dx) * dx_dr((m + 0.5) * dx) * (der - d2r_dx2((m + 0.5) * dx) * first_deriv_comp(m, field));
}
double sixth_deriv(int m, double* field) {
    double der = 0.0;
    
    if (m == 0) {
        return der = pow(dx_dr((m+0.5) * dx),0.0) * (field[m + 2] - 6.0 * field[m + 1] + 15.0 * field[m] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);
    }
    else if (m == 1) {
        return der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m] - 6.0 * field[m - 1] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);

    }
    else if (m == 2) {
       return  der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m - 2] - 6.0 * field[m - 2] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);

    }
    else if (m == lattice_size - 1) {
        return der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m - 6] - 6.0 * field[m - 5] + 15.0 * field[m - 4] - 20.0 * field[m - 3] + 15.0 * field[m - 2] - 6.0 * field[m - 1] + field[m]) / pow(dx, 6.0);
    }
    else if (m == lattice_size - 2) {
        return der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m - 5] - 6.0 * field[m - 4] + 15.0 * field[m - 3] - 20.0 * field[m - 2] + 15.0 * field[m - 1] - 6.0 * field[m] + field[m + 1]) / pow(dx, 6.0);
    }
    else if (m == lattice_size - 3) {   
        return der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m - 4] - 6.0 * field[m - 3] + 15.0 * field[m - 2] - 20.0 * field[m - 1] + 15.0 * field[m] - 6.0 * field[m + 1] + field[m + 2]) / pow(dx, 6.0);
    }
    else {

        return der = pow(dx_dr((m+0.5) * dx), 0.0) * (field[m - 3] - 6.0 * field[m - 2] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);
    }
}
__complex__ double sixth_deriv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;

    if (m == 0) {
        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m + 2] - 6.0 * field[m + 1] + 15.0 * field[m] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);
    }
    else if (m == 1) {
        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m] - 6.0 * field[m - 1] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);

    }
    else if (m == 2) {
        return  der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m - 2] - 6.0 * field[m - 2] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);

    }
    else if (m == lattice_size - 1) {
        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m - 6] - 6.0 * field[m - 5] + 15.0 * field[m - 4] - 20.0 * field[m - 3] + 15.0 * field[m - 2] - 6.0 * field[m - 1] + field[m]) / pow(dx, 6.0);
    }
    else if (m == lattice_size - 2) {
        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m - 5] - 6.0 * field[m - 4] + 15.0 * field[m - 3] - 20.0 * field[m - 2] + 15.0 * field[m - 1] - 6.0 * field[m] + field[m + 1]) / pow(dx, 6.0);
    }
    else if (m == lattice_size - 3) {
        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m - 4] - 6.0 * field[m - 3] + 15.0 * field[m - 2] - 20.0 * field[m - 1] + 15.0 * field[m] - 6.0 * field[m + 1] + field[m + 2]) / pow(dx, 6.0);
    }
    else {

        return der = pow(dx_dr((m + 0.5) * dx), 0.0) * (field[m - 3] - 6.0 * field[m - 2] + 15.0 * field[m - 1] - 20.0 * field[m] + 15.0 * field[m + 1] - 6.0 * field[m + 2] + field[m + 3]) / pow(dx, 6.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Matrix multiplication, dot product and setting initial matrices */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void matrix_mult(double** matrix1, double** matrix2, double** mult) {

    // Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
    for (int i = 0; i < lattice_size; ++i)
    {
        for (int j = 0; j < lattice_size; ++j)
        {
            for (int k = 0; k < lattice_size; ++k)
            {
                mult[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}
void dot_product(double** matrix, double* field, double* product) {
    //#pragma omp parallel for
    for (int i = 0; i < lattice_size; i++) {
        product[i] = 0.0;
        for (int j = 0; j < lattice_size; j++) {
            product[i] = product[i] + matrix[i][j] * field[j];
        }
    }
}
void set_der_mats(double** first_der, double** second_der, double** diag) {
    double x[lattice_size], r;
    make_points(x);
    //printf("\n %.10f \n", -dx_dr(x[0]) * 0.5 / (dx));
    for (int i = 0; i < lattice_size; i++) {

            if (i == 0) {
                first_der[i][i]     = -8.0 * dx_dr(x[i]) / (12.0 * dx);
                first_der[i][i + 1] =  9.0 * dx_dr(x[i]) / (12.0 * dx);
                first_der[i][i + 2] = -1.0 * dx_dr(x[i]) / (12.0 * dx);

                second_der[i][i]     = dx_dr(x[i]) * dx_dr(x[i]) * (-14.0 / (12.0 * dx * dx)  -d2r_dx2(x[i]) * first_der[i][i]);
                second_der[i][i + 1] = dx_dr(x[i]) * dx_dr(x[i]) * ( 15.0 / (12.0 * dx * dx)  -d2r_dx2(x[i]) * first_der[i][i + 1]);
                second_der[i][i + 2] = dx_dr(x[i]) * dx_dr(x[i]) * (-1.0  / (12.0 * dx * dx) - d2r_dx2(x[i]) * first_der[i][i + 2]);
            }
            else if (i == 1) {
                first_der[i][i - 1] = -7.0 * dx_dr(x[i]) / (12.0 * dx);
                first_der[i][i + 1] =  8.0 * dx_dr(x[i]) / (12.0 * dx);
                first_der[i][i + 2] = -1.0 * dx_dr(x[i]) / (12.0 * dx);

                second_der[i][i - 1] = dx_dr(x[i]) * dx_dr(x[i]) * ( 15.0 / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 1]);
                second_der[i][i]     = dx_dr(x[i]) * dx_dr(x[i]) * (-30.0 / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i]);
                second_der[i][i + 1] = dx_dr(x[i]) * dx_dr(x[i]) * ( 16.0 / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i + 1]);
                second_der[i][i + 2] = dx_dr(x[i]) * dx_dr(x[i]) * (-1.0  / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i + 2]);
            }
            else if (i == lattice_size - 2) {
                first_der[i][i-2] =  dx_dr(x[i]) * 1.0 / (6.0*dx);
                first_der[i][i-1] = -dx_dr(x[i]) * 6.0 / (6.0*dx);
                first_der[i][i]   =  dx_dr(x[i]) * 3.0 / (6.0*dx);
                first_der[i][i+1] =  dx_dr(x[i]) * 2.0 / (6.0*dx);

                second_der[i][i - 2] = dx_dr(x[i]) * dx_dr(x[i]) * (0.0 /(dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 2]);
                second_der[i][i - 1] = dx_dr(x[i]) * dx_dr(x[i]) *( 1.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 1]);
                second_der[i][i]     = dx_dr(x[i]) * dx_dr(x[i]) *(-2.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i]);
                second_der[i][i + 1] = dx_dr(x[i]) * dx_dr(x[i]) *( 1.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i + 1]);
            }
            else if (i == lattice_size - 1) {
                first_der[i][i-2] =  dx_dr(x[i]) * 1.0 / (2.0*dx);
                first_der[i][i-1] = -dx_dr(x[i]) * 4.0 / (2.0*dx);
                first_der[i][i]   =  dx_dr(x[i]) * 3.0 / (2.0*dx);

                second_der[i][i - 2] = dx_dr(x[i]) * dx_dr(x[i]) *( 1.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 2]);
                second_der[i][i - 1] = dx_dr(x[i]) * dx_dr(x[i]) *(-2.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 1]);
                second_der[i][i]     = dx_dr(x[i]) * dx_dr(x[i]) *( 1.0/ (dx * dx) -d2r_dx2(x[i]) * first_der[i][i]);
            }
            else {
                first_der[i][i - 2]  =  dx_dr(x[i]) * 1.0 / (12.0*dx);
                first_der[i][i - 1]  = -dx_dr(x[i]) * 8.0 / (12.0*dx);
                first_der[i][i + 1]  =  dx_dr(x[i]) * 8.0 / (12.0*dx);
                first_der[i][i + 2]  = -dx_dr(x[i]) * 1.0 / (12.0*dx);

                second_der[i][i - 2] = dx_dr(x[i]) * dx_dr(x[i]) *(-1.0 / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 2]);
                second_der[i][i - 1] = dx_dr(x[i]) * dx_dr(x[i]) *( 16.0/ (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i - 1]);
                second_der[i][i]     = dx_dr(x[i]) * dx_dr(x[i]) *(-30.0/ (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i]);
                second_der[i][i + 1] = dx_dr(x[i]) * dx_dr(x[i]) *( 16.0/ (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i + 1]);
                second_der[i][i + 2] = dx_dr(x[i]) * dx_dr(x[i]) *(-1.0 / (12.0 * dx * dx) -d2r_dx2(x[i]) * first_der[i][i + 2]);

            }

            diag[i][i] = 1.0;
        
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* This function provides a version of gsl's Bessel function that ignores any underflow error */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double gsl_sf_bessel_jl_safe(int l, double x) {
    gsl_sf_result answer;
    gsl_error_handler_t* old_error_handler = gsl_set_error_handler_off();    // turn off the error handler
    int error_code = gsl_sf_bessel_jl_e(l, x, &answer);                       //compute the answer, and construct an error code
    gsl_set_error_handler(old_error_handler); //reset the error handler
    if (error_code == GSL_SUCCESS) {                                                  //if there's no error then return the correct answer
        return answer.val;
    }
    else {
        //printf ("error in gsl_sf_bessel_jl_safe: %s\n", gsl_strerror (error_code));
        //exit(1);
        if (error_code == GSL_EUNDRFLW) {
            return 0.0;
        }
        else {
            printf("error in gsl_sf_bessel_jl_safe: %s\n", gsl_strerror(error_code));
            exit(1);
        }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* This function provides the initial profile functions */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double phi_mode_profile_0(double k, int l, double r) {
    return (sqrt(k / PI) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l));
}
//---
double phi_mode_profile_0_prime(double k, int l, double r) {
    return (-k * sqrt(k / PI) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l));
}
//---
double phi_mode_profile_massive(double msq, double k, int l, double r) {
    return (k / sqrt(PI * sqrt(k * k + msq)) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l));
}
//---
double phi_mode_profile_massive_prime(double msq, double k, int l, double r) {
    return (-k * k / sqrt(PI * sqrt(k * k + msq)) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l));
}
void initial_conditions(Classical_fields* c_fields, Metric_Fields* metric) {
    double x[lattice_size];
    make_points(x);

    /* METRIC FIELDS */
    for (int i = 0; i < lattice_size; ++i) {
        metric->alpha[i]       = 1.0;
        metric->psi[i]         = 1.0;
        metric->beta[i]        = -1.0 / r_max;
        metric->K_hat[i]       = 0.0;
    }

    /* CLASSICAL MATTER FIELDS */
    for (int i = 0; i < lattice_size; ++i) {
        double r, phi, phi_prime;

        r = r_x(x[i]);

        phi       = amplitude * exp(-1.0 / 2.0 * pow((r - initial_radius) / initial_width, 2.0));
        phi_prime = -(r - initial_radius) / pow(initial_width, 2.0) * phi;

        /* CHI */
        c_fields->chi[i] = -amplitude / (initial_width * initial_width) * exp(-r * r / (2.0*initial_width * initial_width));
        //c_fields->chi[i] = phi_prime / r;

        /* PI */
        c_fields->pi[i] = 0.0;  
        //c_fields->pi[i] = phi / r + phi_prime;
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the quantum variables */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initial_conditions_quantum(Classical_fields* c_fields, Quantum_fields** q_fields, Metric_Fields* metric) {

    /* QUANTUM MATTER FIELDS */
    double x[lattice_size];
    make_points(x);
    //the initial data for the quantum vacuum modes phi
    for (int i = 0; i < lattice_size; ++i) {
        double k_wavenumber, omega_phi;
        int l_value;
        double r;
        r = r_x(x[i]);
        for (int k = 0; k < number_of_k_modes; ++k) {
            k_wavenumber = (k_start + (k + 1)) * k_min;
            for (int l = 0; l < number_of_l_modes; ++l) {
                l_value = l_start + l * l_step;
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {//cycle through the quantum fields and initialize them
                    omega_phi = sqrt(k_wavenumber * k_wavenumber + massSq[which_q_field]);
                    if (massSq[which_q_field] == 0) {
                        q_fields[which_q_field]->phi[k][l][i] = phi_mode_profile_0(k_wavenumber, l_value, r);                 //set the r!=0 zero values
                        q_fields[which_q_field]->chi[k][l][i] = phi_mode_profile_0_prime(k_wavenumber, l_value, r); //this is a place-holder for phi_prime, it will get replaced with psi
                        
                    }
                    else {
                        q_fields[which_q_field]->phi[k][l][i] = phi_mode_profile_massive(massSq[which_q_field], k_wavenumber, l_value, r);
                        q_fields[which_q_field]->chi[k][l][i] = phi_mode_profile_massive_prime(massSq[which_q_field], k_wavenumber, l_value, r); //this is a place-holder for phi_prime, it will get replaced with psi
                        

                    }
                        
                    //then sort out the momenta
                    q_fields[which_q_field]->pi[k][l][i] = -I * omega_phi * q_fields[which_q_field]->phi[k][l][i];
                    
                }

            }   
        }
    }
}

void initialise_quantum(Classical_fields* c_fields, Quantum_fields** q_fields, Metric_Fields* metric) {

    /* QUANTUM MATTER FIELDS */
    double x[lattice_size];
    make_points(x);
    for (int i = 0; i < lattice_size; ++i) {
        double k_wavenumber, omega_phi;
        int l_value;
        double r;
        r = r_x(x[i]);
        for (int k = 0; k < number_of_k_modes; ++k) {
            k_wavenumber = (k_start + (k + 1)) * k_min;
            for (int l = 0; l < number_of_l_modes; ++l) {
                l_value = l_start + l * l_step;
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                    q_fields[which_q_field]->pi[k][l][i] = q_fields[which_q_field]->pi[k][l][i];
                    q_fields[which_q_field]->chi[k][l][i] = 1.0 /r * q_fields[which_q_field]->chi[k][l][i];
                }

            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the norm */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double norm(__complex__ double number) {
    double nor = 0.0;
    nor = (pow((__real__ number), 2.0) + pow((__imag__ number), 2.0));
    return nor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the bilinears */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears(int i, Bi_Linears* bi_linears, Classical_fields* c_fields, Quantum_fields** q_fields, Metric_Fields* metric, double cosm_const) {
    double r, r_l;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    double psi, beta, alpha;
    int l_value;

    double x[lattice_size];
    make_points(x);
    r = r_x(x[i]);
    alpha = metric->alpha[i];
    beta = metric->beta[i];
    psi = metric->psi[i];

    phi_phi = 0.0;
    chi_chi = 0.0;
    pi_pi = 0.0;
    chi_pi = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if (coherent_state_switch != 0) {
        //phi_phi = c_fields->phi[i] * c_fields->phi[i];
        chi_chi = c_fields->chi[i] * c_fields->chi[i];
        pi_pi = c_fields->pi[i] * c_fields->pi[i];
        chi_pi = c_fields->chi[i] * c_fields->pi[i];
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    }

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = r^l pi
    //Psi = lr^{l-1} u + r^l psi
    if (hbar != 0) {
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;
                    r_l = pow(r, l_value);


                    /* PHI MODE */
                    Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);

                    /* CHI MODE */
                    Chi_mode = l_value * pow(metric->psi[i],2.0) * pow(r, l_value - 2) * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->chi[k][l][i]);

                    /* PI MODE */
                    Pi_mode = r_l * q_fields[which_q_field]->pi[k][l][i];


                    /* ACTUAL BILINEARS */
                    phi_phi                               = phi_phi                               + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi                               = chi_chi                               + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi                                 = pi_pi                                 + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi                                = chi_pi                                + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * (__real__(Pi_mode * conj(Chi_mode)));
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);



                }
            }
        }
    }

    bi_linears->phi_phi     = phi_phi;
    bi_linears->chi_chi_rr  = chi_chi * r * r;
    bi_linears->pi_pi       = pi_pi; 
    bi_linears->chi_pi      = chi_pi;
    bi_linears->del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the cosmological constant */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double set_cosm_constant(Classical_fields* c_fields, Quantum_fields** q_fields, Metric_Fields* metric) {

    int i = 0;

    double r, r_l;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    double psi, beta, alpha;
    int l_value;

    double x[lattice_size];
    make_points(x);
    r = r_x(x[i]);
    alpha = metric->alpha[i];
    beta = metric->beta[i];
    psi = metric->psi[i];

    phi_phi = 0.0;
    chi_chi = 0.0;
    pi_pi = 0.0;
    chi_pi = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = r^l pi
    //Psi = lr^{l-1} u + r^l psi
    //if (hbar != 0) {
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;
                    r_l = pow(r, l_value);


                    /* PHI MODE */
                    Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);

                    /* CHI MODE */
                    Chi_mode = l_value * pow(metric->psi[i], 2.0) * pow(r, l_value - 2) * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->chi[k][l][i]);

                    /* PI MODE */
                    Pi_mode = r_l * q_fields[which_q_field]->pi[k][l][i];


                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi   = pi_pi   + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi  = chi_pi  + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * (__real__(Pi_mode * conj(Chi_mode)));
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);



                }
            }
        }
    //}

        return del_theta_phi_del_theta_phi_over_r_sq;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the initial psi, uses a fourth order matrix method */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void solve_psi_newton0(double** first_der, double** second_der, double** diag, Metric_Fields* metric, dMetric_Bound_Dot* metric_bound, Classical_fields* c_fields, Quantum_fields** q_fields, double cosm_const) {

    gsl_vector* fa = gsl_vector_alloc(lattice_size);
    gsl_vector* fb = gsl_vector_alloc(lattice_size);
    gsl_vector* fc = gsl_vector_alloc(lattice_size);

    gsl_vector* delta_psi = gsl_vector_alloc(lattice_size);
    gsl_vector_uint* piv = gsl_vector_uint_alloc(lattice_size);
    gsl_matrix* mat = gsl_matrix_calloc(lattice_size, 7);

    double w = 0.99;
    double eps;
    double x[lattice_size];
    double max_delta = 0.0;
    make_points(x);

    for (int i = 0; i < lattice_size; i++) {
        gsl_vector_set(delta_psi, i, 0.0);
        metric->psi[i] = metric->psi[i] * metric_bound->psi / metric->psi[lattice_size - 1];
    }
    int k = 0;
    double deltapsi0 = 0.1;
    for (int k = 0; k < 10; k++) {
        for (int i = 0; i < lattice_size; i++) {
            double r, Khat, psi, psi_prime, pi, chi, psi_prime_prime;
            double pi_pi, chi_chi, delthetaphisquared, cc;

            pi_pi = c_fields->pi[i] * c_fields->pi[i];
            chi_chi = c_fields->chi[i] * c_fields->chi[i]*r*r;
            delthetaphisquared = 0.0;
            cc = 0.0;

            r = r_x(x[i]);
            Khat = metric->K_hat[i];
            psi = metric->psi[i];
            psi_prime = first_deriv(i, metric->psi);
            psi_prime_prime = second_deriv(i, metric->psi);
            pi = c_fields->pi[i];
            chi = c_fields->chi[i];
             
            gsl_vector_set(fa, i, 2.0 / r);
            gsl_vector_set(fb, i, -21.0 / 16.0 * pow(psi, -8) * pow(r, 4) * Khat * Khat - 3.0 / 8.0 * (M_P * M_P) * pow(psi, -4.0) * (pi_pi + chi_chi + 2.0 * delthetaphisquared -  2.0 * exp(-pow(r * metric->psi[i] * metric->psi[i] / r_damp, 20.0)) * cc));
            gsl_vector_set(fc, i, -psi_prime_prime - 2.0 * psi_prime / r - 3.0 / 16.0 * pow(psi, -7.0) * pow(r, 4.0) * Khat * Khat - 1.0 / 8.0 * (M_P * M_P) * pow(psi, -3.0) * (pi_pi + chi_chi + 2.0 * delthetaphisquared - 2.0 * exp(-pow(r * metric->psi[i] * metric->psi[i] / r_damp, 20.0)) * cc));

        }

        for (int i = 0; i < lattice_size; i++) {
            double r;
            r = r_x(x[i]);
            if (i == 0) {
                for (int j = 0; j < 3; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == 1) {
                for (int j = 0; j < 4; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == lattice_size - 1) {
                for (int j = lattice_size - 2; j < lattice_size; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == lattice_size - 2) {
                for (int j = lattice_size - 4; j < lattice_size; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else {
                for (int j = i - 2; j < i + 3; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
        }
        gsl_matrix_set(mat, lattice_size - 1, 4, 1.0);
        gsl_matrix_set(mat, lattice_size - 2, 5, 0.0);
        gsl_matrix_set(mat, lattice_size - 3, 6, 0.0);

        gsl_vector_set(fc, lattice_size - 1, 0.0);

        gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat, piv);
        gsl_linalg_LU_band_solve(2, 2, mat, piv, fc, delta_psi);

        for (int i = 0; i < lattice_size; i++) {
            metric->psi[i] = metric->psi[i] + w * gsl_vector_get(delta_psi, i);
        }
        deltapsi0 = gsl_vector_get(delta_psi, 0) / metric->psi[0];

    }
    gsl_vector_free(fa);
    gsl_vector_free(fb);
    gsl_vector_free(fc);
    gsl_vector_free(delta_psi);
    gsl_vector_uint_free(piv);
    gsl_matrix_free(mat);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates psi, uses a fourth order matrix method */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void solve_psi_newton(double** first_der, double** second_der, double** diag, Metric_Fields* metric, dMetric_Bound_Dot* metric_bound, Classical_fields* c_fields, Quantum_fields** q_fields, double cosm_const) {

    gsl_vector* fa = gsl_vector_alloc(lattice_size);
    gsl_vector* fb = gsl_vector_alloc(lattice_size);
    gsl_vector* fc = gsl_vector_alloc(lattice_size);

    gsl_vector* delta_psi = gsl_vector_alloc(lattice_size);
    gsl_vector_uint* piv  = gsl_vector_uint_alloc(lattice_size);
    gsl_matrix* mat       = gsl_matrix_calloc(lattice_size, 7);

    double w = 0.99;
    double eps;
    double x[lattice_size];
    double max_delta = 0.0;
    make_points(x);
    for (int i = 0; i < lattice_size; i++) {
        gsl_vector_set(delta_psi, i, 0.0);
        metric->psi[i] = metric->psi[i]*metric_bound->psi/metric->psi[lattice_size-1];
    }
    int k=0;
    double deltapsi0=0.1;
    for (int k = 0; k < 10; k++) {
        for (int i = 0; i < lattice_size; i++) {
            double r, Khat, psi, psi_prime, pi, chi, psi_prime_prime;
            double pi_pi, chi_chi, delthetaphisquared;
            r               = r_x(x[i]);
            Bi_Linears    bi_linears;
            set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, cosm_const);
            pi_pi = bi_linears.pi_pi;
            chi_chi = bi_linears.chi_chi_rr;
            delthetaphisquared = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;
            //pi_pi = c_fields->pi[i] * c_fields->pi[i];
            //chi_chi = c_fields->chi[i] * c_fields->chi[i]*r*r;
            //delthetaphisquared = 0.0;

            Khat            = metric->K_hat[i];
            psi             = metric->psi[i];
            psi_prime       = first_deriv(i,  metric->psi);
            psi_prime_prime = second_deriv(i, metric->psi);
            pi              = c_fields->pi[i];
            chi             = c_fields->chi[i];

            gsl_vector_set(fa, i, 2.0 / r);
            gsl_vector_set(fb, i, -21.0 / 16.0 * pow(psi, -8) * pow(r, 4) * Khat * Khat - 3.0 / 8.0 * (M_P * M_P) * pow(psi, -4.0) * (pi_pi + chi_chi + 2.0 * pow(psi, 4.0) * delthetaphisquared - pow(psi, 8.0) * 2.0* cosm_const));
            gsl_vector_set(fc, i, -psi_prime_prime - 2.0 * psi_prime / r - 3.0 / 16.0 * pow(psi, -7.0) * pow(r, 4.0) * Khat * Khat - 1.0 / 8.0 * (M_P * M_P) * pow(psi, -3.0) * (pi_pi + chi_chi + 2.0*pow(psi,4.0)* delthetaphisquared - pow(psi, 8.0) *2.0* cosm_const));

        }
        for (int i = 0; i < lattice_size; i++) {
            double r;
            r = r_x(x[i]);
            if (i == 0) {
                for (int j = 0; j < 3; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == 1) {
                for (int j = 0; j < 4; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == lattice_size - 1) {
                for (int j = lattice_size - 2; j < lattice_size; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else if (i == lattice_size - 2) {
                for (int j = lattice_size - 4; j < lattice_size; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
            else {
                for (int j = i - 2; j < i + 3; j++) {
                    gsl_matrix_set(mat, j, i - j + 4, second_der[i][j] + gsl_vector_get(fa, i) * first_der[i][j] + gsl_vector_get(fb, i) * diag[i][j]);
                }
            }
        }
        gsl_matrix_set(mat, lattice_size - 1, 4, 1.0);
        gsl_matrix_set(mat, lattice_size - 2, 5, 0.0);
        gsl_matrix_set(mat, lattice_size - 3, 6, 0.0);

        gsl_vector_set(fc, lattice_size - 1, 0.0);

        gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat, piv);
        gsl_linalg_LU_band_solve(2, 2, mat, piv, fc, delta_psi);
        for (int i = 0; i < lattice_size; i++) {
            metric->psi[i] = metric->psi[i] + w*gsl_vector_get(delta_psi,i);
        }
        deltapsi0 = gsl_vector_get(delta_psi, 0)/metric->psi[0];

    }
    gsl_vector_free(fa);
    gsl_vector_free(fb);
    gsl_vector_free(fc);
    gsl_vector_free(delta_psi);
    gsl_vector_uint_free(piv);
    gsl_matrix_free(mat);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises all the metric functions (including psi) */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initialise_metric(double** first_der, double** second_der, double** diag, Metric_Fields* metric, Classical_fields* c_fields, Quantum_fields** q_fields, double cosm_const) {

    double x[lattice_size];
    make_points(x);
    double cc2[lattice_size];
    gsl_vector* cc = gsl_vector_alloc(lattice_size);
    gsl_vector* sol = gsl_vector_alloc(lattice_size);

    gsl_vector_uint* piv = gsl_vector_uint_alloc(lattice_size);
    gsl_matrix* mat2 = gsl_matrix_calloc(lattice_size, 7);

    dMetric_Bound_Dot dmetric_bound;
    dmetric_bound.alpha = 1.0;
    dmetric_bound.psi = 1.0;
    dmetric_bound.Khat = 0.0;
    dmetric_bound.beta = -nu / r_max;

    // first set K_hat

    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        if (i == 0) {
            for (int j = i; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j] * r + 5.0 * diag[i][j]);
            }
        }
        else if (i == 1) {
            for (int j = i - 1; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j] * r + 5.0 * diag[i][j]);
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = i - 2; j < i + 2; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j] * r + 5.0 * diag[i][j]);
            }
        }
        else if (i == lattice_size - 1) {
            for (int j = i - 2; j < i + 1; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j] * r + 5.0 * diag[i][j]);
            }
        }
        else {
            for (int j = i - 2; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j] * r + 5.0 * diag[i][j]);
            }
        }
        gsl_vector_set(cc, i, -(M_P * M_P) * c_fields->chi[i]*c_fields->pi[i]);
    }

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);

    for (int i = 0; i < lattice_size; i++) {
        metric->K_hat[i] = gsl_vector_get(sol, i);
    }

    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);

    // now set psi
    solve_psi_newton0(first_der, second_der, diag, metric, &dmetric_bound, c_fields, q_fields, cosm_const);
    

    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);
    // now set alpha
    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        double pi_pi;
        Bi_Linears    bi_linears;
        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, cosm_const);
        //pi_pi = bi_linears.pi_pi + pow(metric->psi[i], 8.0)*cosm_const;
        pi_pi = c_fields->pi[i] * c_fields->pi[i];
        if (i == 0) {
            for (int j = 0; j < 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == 1) {
            for (int j = 0; j < 4; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == lattice_size - 1) {
            for (int j = lattice_size - 2; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = lattice_size - 4; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else {
            for (int j = i - 2; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        gsl_vector_set(cc, i, 0.0);
    }
    gsl_matrix_set(mat2, lattice_size - 1, 4, 1.0);
    gsl_matrix_set(mat2, lattice_size - 2, 5, 0.0);
    gsl_matrix_set(mat2, lattice_size - 3, 6, 0.0);

    gsl_vector_set(cc, lattice_size - 1, dmetric_bound.alpha);

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);
    for (int i = 0; i < lattice_size; i++) {
        metric->alpha[i] = gsl_vector_get(sol, i);
    }


    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);

    // now set beta
    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        if (i == 0) {
            for (int j = 0; j < 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == 1) {
            for (int j = 0; j < 4; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == lattice_size - 1) {
            for (int j = lattice_size - 2; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = lattice_size - 4; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else {
            for (int j = i - 2; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        gsl_vector_set(cc, i, 3.0 / 2.0 * r * metric->alpha[i] * pow(metric->psi[i], -6.0) * metric->K_hat[i]);
    }

    gsl_matrix_set(mat2, lattice_size - 1, 4, 1.0);
    gsl_matrix_set(mat2, lattice_size - 2, 5, 0.0);
    gsl_matrix_set(mat2, lattice_size - 3, 6, 0.0);
    dmetric_bound.beta = -nu / r_max * pow(metric->psi[lattice_size - 1], -2.0) * metric->alpha[lattice_size - 1];
    gsl_vector_set(cc, lattice_size - 1, dmetric_bound.beta);

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);
    for (int i = 0; i < lattice_size; i++) {
        metric->beta[i] = gsl_vector_get(sol, i);
    }
    gsl_vector_free(cc);
    gsl_vector_free(sol);
    gsl_vector_uint_free(piv);
    gsl_matrix_free(mat2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates all the metric functions at each time step */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_K_psi_alpha_beta(double** first_der, double** second_der, double** diag, Metric_Fields* metric, dMetric_Bound_Dot* metric_bound, Classical_fields* c_fields, Quantum_fields** q_fields, double cosm_const){
    
    double x[lattice_size];
    make_points(x);
    double cc2[lattice_size];
    gsl_vector* cc = gsl_vector_alloc(lattice_size);
    gsl_vector* sol = gsl_vector_alloc(lattice_size);

    gsl_vector_uint* piv = gsl_vector_uint_alloc(lattice_size);
    gsl_matrix* mat2 = gsl_matrix_calloc(lattice_size,7);

    // first set K_hat
    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        double pi_chi;
        Bi_Linears    bi_linears;
        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, cosm_const);
        //pi_chi = bi_linears.chi_pi;
        pi_chi = c_fields->pi[i] * c_fields->chi[i];
        if (i == 0) {
            for (int j = i; j < i+3; j++) {
                gsl_matrix_set(mat2,j,i - j + 4, first_der[i][j]*r + 5.0 * diag[i][j]); 
            }
        }
        else if (i == 1) {
            for (int j = i-1; j < i+3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]*r + 5.0 * diag[i][j]);
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = i-2; j < i+2; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]*r + 5.0 * diag[i][j]);
            }
        }
        else if (i == lattice_size-1) {
            for (int j = i - 2; j < i + 1; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]*r + 5.0 * diag[i][j]);
            }
        }
        else {
            for (int j = i-2; j < i+3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]*r + 5.0 * diag[i][j]);
            }
        }
        gsl_vector_set(cc,i, -(M_P*M_P)*pi_chi);
    }

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);
    for (int i = 0; i < lattice_size; i++) {
        metric->K_hat[i] = gsl_vector_get(sol, i);
    }

    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);
    
    // now set psi
    solve_psi_newton(first_der, second_der, diag, metric, metric_bound, c_fields, q_fields, cosm_const);

    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);

    // now set alpha
    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        double pi_pi;
        Bi_Linears    bi_linears;
        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, cosm_const);
        //pi_pi = bi_linears.pi_pi + cosm_const;
        pi_pi = c_fields->pi[i] * c_fields->pi[i];
        if (i == 0) {   
            for (int j = 0; j < 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == 1) {
            for (int j = 0; j < 4; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == lattice_size - 1) {
            for (int j = lattice_size - 2; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = lattice_size - 4; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        else {
            for (int j = i - 2; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, (second_der[i][j] + 2.0 * first_der[i][j] * (1.0 / r + first_deriv(i, metric->psi) / metric->psi[i])
                    - diag[i][j] * ((M_P * M_P) * pow(metric->psi[i], -4.0) * pi_pi + 3.0 / 2.0 * pow(metric->psi[i], -8.0) * pow(r * r * metric->K_hat[i], 2.0))));
            }
        }
        gsl_vector_set(cc, i, 0.0);
    }
    gsl_matrix_set(mat2, lattice_size - 1, 4, 1.0);
    gsl_matrix_set(mat2, lattice_size - 2, 5, 0.0);
    gsl_matrix_set(mat2, lattice_size - 3, 6, 0.0);

    gsl_vector_set(cc, lattice_size - 1, metric_bound->alpha);

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);

    for (int i = 0; i < lattice_size; i++) {
        metric->alpha[i] = gsl_vector_get(sol, i);
    }
    
    
    gsl_vector_uint_set_zero(piv);
    gsl_vector_set_zero(cc);
    gsl_vector_set_zero(sol);
    gsl_matrix_set_zero(mat2);

    // now set beta
    for (int i = 0; i < lattice_size; i++) {
        double r;
        r = r_x(x[i]);
        if (i == 0) {
            for (int j = 0; j < 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == 1) {
            for (int j = 0; j < 4; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == lattice_size - 1) {
            for (int j = lattice_size - 2; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else if (i == lattice_size - 2) {
            for (int j = lattice_size - 4; j < lattice_size; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        else {
            for (int j = i - 2; j < i + 3; j++) {
                gsl_matrix_set(mat2, j, i - j + 4, first_der[i][j]);
            }
        }
        gsl_vector_set(cc,i, 3.0 / 2.0 * r * metric->alpha[i] * pow(metric->psi[i], -6.0) * metric->K_hat[i]);
    }

    gsl_matrix_set(mat2,lattice_size - 1, 4, 1.0);
    gsl_matrix_set(mat2,lattice_size - 2, 5, 0.0);
    gsl_matrix_set(mat2,lattice_size - 3, 6, 0.0);
    metric_bound->beta = -nu / r_max * pow(metric->psi[lattice_size - 1], -2.0) * metric->alpha[lattice_size - 1];
    gsl_vector_set(cc, lattice_size - 1, metric_bound->beta);

    gsl_linalg_LU_band_decomp(lattice_size, 2, 2, mat2, piv);
    gsl_linalg_LU_band_solve(2, 2, mat2, piv, cc, sol);

    for (int i = 0; i < lattice_size; i++) {
        metric->beta[i] = gsl_vector_get(sol, i);
    }
    gsl_vector_free(cc);
    gsl_vector_free(sol);
    gsl_vector_uint_free(piv);
    gsl_matrix_free(mat2);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the psi constraint */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void psi_constraint(int n,double dt, double constr[lattice_size],double psi_n[lattice_size], double psi_nm1[lattice_size], double alpha_nm1[lattice_size], double beta_nm1[lattice_size], double Khat_nm1[lattice_size]) {
    double x[lattice_size];
    make_points(x);
    
    for (int i = 0; i < lattice_size; i++) {
        constr[i] = (psi_n[i] - psi_nm1[i]) / dt -
                                 (r_x(x[i]) * beta_nm1[i] * first_deriv(i, psi_nm1) + 1.0 / 2.0 * beta_nm1[i] * psi_nm1[i] + 1.0 / 4.0 * r_x(x[i]) * r_x(x[i]) * alpha_nm1[i] * pow(psi_nm1[i], -5) * Khat_nm1[i]);
        constr[i] = constr[i];// / ((psi_n[i] - psi_nm1[i]) / dt);

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the K constraint */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void K_constraint(int n, double dt, double constr[lattice_size], double K_n[lattice_size], double K_nm1[lattice_size], double psi_nm1[lattice_size], double alpha_nm1[lattice_size], double beta_nm1[lattice_size], double chi_nm1[lattice_size]) {
    double r, beta, alpha, alpha_prime, psi,psi_prime, K_hat, chi, alpha_primeprime, psi_primeprime, Khat_prime;
    double x[lattice_size];
    make_points(x);

    for (int i = 0; i < lattice_size; i++) {
        r                = r_x(x[i]);
        beta             = beta_nm1[i];
        alpha            = alpha_nm1[i];
        alpha_prime      = first_deriv(i, alpha_nm1);
        alpha_primeprime = second_deriv(i, alpha_nm1);
        psi              = psi_nm1[i];
        psi_prime        = first_deriv(i, psi_nm1);
        psi_primeprime   = second_deriv(i, psi_nm1);
        K_hat            = K_nm1[i];
        Khat_prime       = first_deriv(i, K_nm1);
        chi              = chi_nm1[i]; 


        constr[i] = (K_n[i] - K_nm1[i]) / dt -
            (r * beta * Khat_prime + 5.0 * beta * K_hat + 1.5 * alpha * r * r * pow(psi, -6) * K_hat * K_hat
            - 2.0 / 3.0 * psi * psi * ((alpha_primeprime - alpha_prime / r) / (r * r))
            - 4.0 / 3.0 * alpha * psi * ((psi_primeprime - psi_prime / r) / (r * r))
            + 4.0 / (r * r) * psi_prime * (alpha * psi_prime + 2.0 / 3.0 * psi * alpha_prime)
            - 2.0 / 3.0 * (M_P) * (M_P) * alpha * pow(psi, -2) * chi * chi);

        constr[i] = constr[i];// / ((K_n[i] - K_nm1[i]) / dt);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the time derivative (right hand side) for all variables evolved in time */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void df_dt(double** sixth_der, double dt, Metric_Fields* metric, dMetric_Bound_Dot* bound, dMetric_Bound_Dot* bound_dot, Classical_fields* c_fields, Classical_fields* c_fields_dot, Quantum_fields** q_fields, Quantum_fields** q_fields_dot) {

    double x[lattice_size];
    make_points(x);
    for (int i = 0; i < lattice_size; ++i) {

        double r;
        double alpha, alpha_prime, psi, psi_prime, beta, K_hat, pi, chi;
        double alpha_primeprime, psi_primeprime, Khat_prime;
        double chi_prime, pi_prime, beta_prime;

        r = r_x(x[i]);

        alpha       = metric->alpha[i];
        alpha_prime = first_deriv(i, metric->alpha);
        psi         = metric->psi[i];
        psi_prime   = first_deriv(i, metric->psi);
        beta        = metric->beta[i];
        K_hat       = metric->K_hat[i];

        chi        = c_fields->chi[i];
        pi         = c_fields->pi[i];

        chi_prime      = first_deriv(i, c_fields->chi);
        pi_prime       = first_deriv(i, c_fields->pi);

        Khat_prime = first_deriv(i, metric->K_hat);
        beta_prime = first_deriv(i, metric->beta);
        alpha_primeprime = second_deriv(i, metric->alpha);
        psi_primeprime = second_deriv(i, metric->psi);


        
        c_fields_dot->chi[i] = r * beta * chi_prime
            + (3.0 * beta + 2.0 * alpha * pow(psi, -6.0) * r * r * K_hat) * chi
            + alpha * pow(psi, -2.0) * pi_prime / r
            + pow(psi, -3.0) * pi * (psi * alpha_prime - 4.0 * alpha * psi_prime) / r;

        c_fields_dot->pi[i] = r * beta * pi_prime
            + (2.0 * beta + alpha * pow(psi, -6.0) * r * r * K_hat) * pi
            + alpha * pow(psi, -2.0) * r * chi_prime
            + pow(psi, -2.0) * (r * alpha_prime + 3.0 * alpha) * chi;

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                int l_value;
                l_value = l_start + l*l_step;
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
                    __complex__ double phi_mode, chi_mode, pi_mode;
                    __complex__ double phi_mode_der, chi_mode_der, pi_mode_der;

                    phi_mode = q_fields[which_q_field]->phi[k][l][i];
                    chi_mode = q_fields[which_q_field]->chi[k][l][i];
                    pi_mode  = q_fields[which_q_field]->pi[k][l][i];

                    phi_mode_der = first_deriv_comp(i, q_fields[which_q_field]->phi[k][l]);
                    chi_mode_der = first_deriv_comp(i, q_fields[which_q_field]->chi[k][l]);
                    pi_mode_der  = first_deriv_comp(i, q_fields[which_q_field]->pi[k][l]);
                    
                    q_fields_dot[which_q_field]->phi[k][l][i] = (pow(r/psi,2.0)*beta*chi_mode + l_value*beta*phi_mode + alpha*pow(psi,-4.0)*pi_mode);

                    q_fields_dot[which_q_field]->chi[k][l][i] =  r*beta*chi_mode_der + ( (l_value+3.0)*beta + 2.0*alpha*pow(psi,-6.0)*r*r*K_hat)*chi_mode + alpha*pow(psi,-2.0)*pi_mode_der/r
                                                                + pow(psi,-3.0)/r*(psi*alpha_prime-4.0*alpha*psi_prime)*pi_mode + 3.0/2.0*l_value*alpha*pow(psi,-4.0)*K_hat*phi_mode;

                    q_fields_dot[which_q_field]->pi[k][l][i]  = r*beta*pi_mode_der + ( (l_value+2.0)*beta + alpha*pow(psi,-6.0)*r*r*K_hat)*pi_mode + alpha*pow(psi,-2.0)*r*chi_mode_der
                                                                + pow(psi,-2.0)*(r*alpha_prime+(2.0*l_value+3.0)*alpha)*chi_mode + l_value/r*(alpha_prime+2.0*psi_prime/psi)*phi_mode
                                                                - massSq[which_q_field]*alpha*pow(psi,4.0)*phi_mode;

                }
            }
        }

        c_fields_dot->chi[i] = (metric->alpha[i] != 0.0 ? c_fields_dot->chi[i] + exp(-r * r * pow(psi,4)* 0.0 / 1.0) * (epsilon) *pow(dx_dr(x[i]), 1.0) * pow(dx, 6.0) * sixth_deriv(i, c_fields->chi) : c_fields_dot->chi[i]);
        c_fields_dot->pi[i]  = (metric->alpha[i] != 0.0 ? c_fields_dot->pi[i]  + exp(-r * r * pow(psi,4)* 0.0 / 1.0) * (epsilon) *pow(dx_dr(x[i]), 1.0) * pow(dx, 6.0) * sixth_deriv(i, c_fields->pi)  : c_fields_dot->pi[i]);
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                int l_value;
                l_value = l_start + l * l_step;
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                    q_fields_dot[which_q_field]->phi[k][l][i] = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->phi[k][l][i] + exp(-r * r * pow(psi, 4) * 0.0 / 1.0) * ep2 * (epsilon)*pow(dx_dr(x[i]), 1.0) * pow(dx, 6.0) * sixth_deriv_comp(i, q_fields[which_q_field]->phi[k][l]) : q_fields_dot[which_q_field]->phi[k][l][i]);
                    q_fields_dot[which_q_field]->chi[k][l][i] = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->chi[k][l][i] + exp(-r * r * pow(psi, 4) * 0.0 / 1.0) * ep2 * (epsilon)*pow(dx_dr(x[i]), 1.0) * pow(dx, 6.0) * sixth_deriv_comp(i, q_fields[which_q_field]->chi[k][l]) : q_fields_dot[which_q_field]->chi[k][l][i]);
                    q_fields_dot[which_q_field]->pi[k][l][i]  = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->pi[k][l][i]  + exp(-r * r * pow(psi, 4) * 0.0 / 1.0) * ep2 * (epsilon)*pow(dx_dr(x[i]), 1.0) * pow(dx, 6.0) * sixth_deriv_comp(i, q_fields[which_q_field]->pi[k][l])  : q_fields_dot[which_q_field]->pi[k][l][i]);
                }
            }
        }
    }
    double r;
    double alpha, alpha_prime, psi, psi_prime, beta, K_hat, beta_prime;

    r = r_x(x[lattice_size-1]);

    alpha       = metric->alpha[lattice_size - 1]; 
    psi         = metric->psi[lattice_size - 1]; 
    beta        = metric->beta[lattice_size - 1]; 
    K_hat       = metric->K_hat[lattice_size-1];
    psi_prime   = first_deriv(lattice_size - 1, metric->psi);
    alpha_prime = first_deriv(lattice_size - 1, metric->alpha);
    beta_prime  = first_deriv(lattice_size - 1, metric->beta);

    bound_dot->alpha = r * beta * alpha_prime;
    bound_dot->psi   = r * beta * psi_prime   + 1.0 / 2.0 * beta * psi + 1.0 / 4.0 * r * r * alpha * pow(psi, -5) * K_hat;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates dt throughout the simulation (needed because of CFL condition and that the physical space is shrinking) */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double calc_dt(double* dt, Metric_Fields* metric) {
    double r, ri, rim1;
    double vplus[lattice_size], vminus[lattice_size], vmax[lattice_size];
    double x[lattice_size];
    make_points(x);
    double minimum;
    minimum = dx* r_max;
    
    for (int i = 1; i < lattice_size; ++i) {
        ri   = r_x(x[i]);
        rim1 = r_x(x[i-1]);
        vplus[i]  = -ri * metric->beta[i] + pow(metric->psi[i], -2.0) * metric->alpha[i];
        vminus[i] = -ri * metric->beta[i] - pow(metric->psi[i], -2.0) * metric->alpha[i];
        vmax[i]   = (ri-rim1)/fmax(fabs(vplus[i]), fabs(vminus[i]));
        
        if (vmax[i] < minimum)
        {
            minimum = vmax[i];
        }
    }

    return lambda * minimum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that sets the variables zero */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_zero(Classical_fields* c_fields, Quantum_fields** q_fields) {
    for (int i = 0; i < lattice_size; ++i) {
        c_fields->pi[i] = 0.0;
        c_fields->chi[i] = 0.0;

        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                    q_fields[which_q_field]->phi[k][l][i] = 0.0;
                    q_fields[which_q_field]->chi[k][l][i] = 0.0;
                    q_fields[which_q_field]->pi[k][l][i]  = 0.0;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the Hamiltonian constraint (should be at machine accuracy since Hamiltonian constraint is solved almost exactly for psi) */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double ham_constraint(int n, int i, Metric_Fields* metric, Classical_fields* c_fields) {

    double x[lattice_size];
    make_points(x);
    double constr;
    double r;
    double alpha, alpha_prime, psi, psi_prime, beta, K_hat, pi, chi;
    double alpha_primeprime, psi_primeprime, Khat_prime;
    double chi_prime, pi_prime, beta_prime;

    r = r_x(x[i]);

    alpha = metric->alpha[i];
    alpha_prime = first_deriv(i, metric->alpha);
    psi = metric->psi[i];
    psi_prime = first_deriv(i, metric->psi);
    beta = metric->beta[i];
    K_hat = metric->K_hat[i];

    chi = c_fields->chi[i];
    pi = c_fields->pi[i];

    chi_prime = first_deriv(i, c_fields->chi);
    pi_prime = first_deriv(i, c_fields->pi);

    Khat_prime = first_deriv(i, metric->K_hat);
    beta_prime = first_deriv(i, metric->beta);
    alpha_primeprime = second_deriv(i, metric->alpha);
    psi_primeprime = second_deriv(i, metric->psi);

    constr = 3.0 * pow(K_hat * r * r, 2.0) / (4.0 * pow(psi, 12.0)) + r * r * beta * (5.0 * K_hat + r * Khat_prime) / (alpha * pow(psi, 6.0)) + (8.0 * psi_prime + 4.0 * r * psi_primeprime) / (r * pow(psi, 5.0))
        + M_P * M_P * (r * r * beta * chi * pi / (alpha * pow(psi, 6.0)) + (pow(r * chi, 2.0) + pow(pi, 2.0)) / (2.0 * pow(psi, 8.0)));

    return constr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Full evolution of all fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double full_evolution(double** first_der, double** second_der, double** sixth_der, double** diag, 
                    Metric_Fields* metric, Classical_fields* c_fields,    Classical_fields* c_fields_k1, Classical_fields* c_fields_k2,
                                           Classical_fields* c_fields_k3, Classical_fields* c_fields_k4, Classical_fields* c_fields_sum,
                                           Quantum_fields**  q_fields,    Quantum_fields**  q_fields_k1, Quantum_fields** q_fields_k2,
                                           Quantum_fields**  q_fields_k3, Quantum_fields**  q_fields_k4, Quantum_fields** q_fields_sum, double cosm_const, 
                                           double** fieldmat, double** rmat, double** tmat) {

    
    double alphat[evolve_time_int];
    double horizont[evolve_time_int];
    for (int n = 0; n < evolve_time_int; ++n) {
        horizont[n] = 0.0;
    }
    double time[evolve_time_int];

    double x[lattice_size];
    make_points(x);

    dMetric_Bound_Dot dmetric_bound;
    dMetric_Bound_Dot dmetric_bound_dt_k1;
    dMetric_Bound_Dot dmetric_bound_dt_k2;
    dMetric_Bound_Dot dmetric_bound_dt_k3;
    dMetric_Bound_Dot dmetric_bound_dt_k4;
    dMetric_Bound_Dot dmetric_bound_dt_ksum;


    dmetric_bound.alpha       = 1.0;
    dmetric_bound.psi         = 1.0;
    dmetric_bound.Khat        = 0.0;
    dmetric_bound.beta        = -nu / r_max;


    
    double T;
    int nhor = 0;
    for (int n = 0; n < evolve_time_int; ++n) {
        double cc = 0.0;
        double dt;
        dt = calc_dt(&dt, metric);       
        T = T + dt;
        time[n] = T;
        printf("\nn = %d, T = %f, dt = %f", n,T,dt);
        printf(" and R_max = %f", metric->psi[lattice_size - 1] * metric->psi[lattice_size - 1] * r_max);
        
        for (int i = 0; i < lattice_size; ++i) {


            double r;
            double alpha, alpha_prime, psi, psi_prime, beta, K_hat, pi, chi;
            double alpha_primeprime, psi_primeprime, Khat_prime;
            double chi_prime, pi_prime, beta_prime;
            int l_value = l_start;
            r = r_x(x[i]);
            Bi_Linears    bi_linears;
            set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, cc);

            alpha = metric->alpha[i];
            alpha_prime = first_deriv(i, metric->alpha);
            psi = metric->psi[i];
            psi_prime = first_deriv(i, metric->psi);
            beta = metric->beta[i];
            K_hat = metric->K_hat[i];

            chi = c_fields->chi[i];
            pi = c_fields->pi[i];

            chi_prime = first_deriv(i, c_fields->chi);
            pi_prime = first_deriv(i, c_fields->pi);

            Khat_prime = first_deriv(i, metric->K_hat);
            beta_prime = first_deriv(i, metric->beta);
            alpha_primeprime = second_deriv(i, metric->alpha);
            psi_primeprime = second_deriv(i, metric->psi);

            __complex__ double phi_mode = 0.0, chi_mode = 0.0, pi_mode = 0.0;
            __complex__ double phi_mode_der = 0.0, chi_mode_der = 0.0, pi_mode_der = 0.0;

            phi_mode = q_fields[0]->phi[0][0][i];
            chi_mode = q_fields[0]->chi[0][0][i];
            pi_mode  = q_fields[0]->pi[0][0][i];

            phi_mode_der = first_deriv_comp(i, q_fields[0]->phi[0][0]);
            chi_mode_der = first_deriv_comp(i, q_fields[0]->chi[0][0]);
            pi_mode_der = first_deriv_comp(i, q_fields[0]->pi[0][0]);

            if (n == 0) {
                tmat[0][i] = 0.0;   
                rmat[0][i] = psi * psi * r;
            }
            if (n != evolve_time_int -1){
                tmat[n+1][i] = tmat[n][i] + (alpha )* dt;
                rmat[n + 1][i] = psi * psi * r;
            }

            fieldmat[n][i] =  (bi_linears.phi_phi);
        }

        set_zero(c_fields_k1, q_fields_k1);
        set_zero(c_fields_k2, q_fields_k2);
        set_zero(c_fields_k3, q_fields_k3);
        set_zero(c_fields_k4, q_fields_k4);
        set_zero(c_fields_sum, q_fields_sum);

        dmetric_bound_dt_k1.alpha       = 0.0;
        dmetric_bound_dt_k1.psi         = 0.0;

        dmetric_bound_dt_k2.alpha       = 0.0;
        dmetric_bound_dt_k2.psi         = 0.0;

        dmetric_bound_dt_k3.alpha       = 0.0;
        dmetric_bound_dt_k3.psi         = 0.0;

        dmetric_bound_dt_k4.alpha       = 0.0;
        dmetric_bound_dt_k4.psi         = 0.0;

//#pragma omp parallel for
        for (int i = 0; i < lattice_size; ++i) {
            c_fields_sum->pi[i]  = c_fields->pi[i]; 
            c_fields_sum->chi[i] = c_fields->chi[i]; 

            for(int k=0; k<number_of_k_modes; ++k){
                for(int l=0; l<number_of_l_modes; ++l){
                    for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                        q_fields_sum[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i];
                        q_fields_sum[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i];
                        q_fields_sum[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i];
                    }
                }
            }
        }
        dmetric_bound_dt_ksum.alpha       = dmetric_bound.alpha;
        dmetric_bound_dt_ksum.psi         = dmetric_bound.psi;



        df_dt(sixth_der, dt, metric, &dmetric_bound_dt_ksum, &dmetric_bound_dt_k1, c_fields_sum, c_fields_k1, q_fields_sum, q_fields_k1);
//#pragma omp parallel for
        for (int i = 0; i < lattice_size; ++i) {
            c_fields_sum->pi[i]  = c_fields->pi[i]   + dt * 0.5 * c_fields_k1->pi[i];
            c_fields_sum->chi[i] = c_fields->chi[i]  + dt * 0.5 * c_fields_k1->chi[i];

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields_sum[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt * 0.5 * q_fields_k1[which_q_field]->phi[k][l][i];
                        q_fields_sum[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i] + dt * 0.5 * q_fields_k1[which_q_field]->chi[k][l][i];
                        q_fields_sum[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt * 0.5 * q_fields_k1[which_q_field]->pi[k][l][i];
                    }
                }
            }

        }
        dmetric_bound_dt_ksum.alpha       = dmetric_bound.alpha        + dt * 0.5 * dmetric_bound_dt_k1.alpha;
        dmetric_bound_dt_ksum.psi         = dmetric_bound.psi          + dt * 0.5 * dmetric_bound_dt_k1.psi;


        set_K_psi_alpha_beta(first_der, second_der, diag, metric, &dmetric_bound_dt_ksum, c_fields_sum, q_fields_sum, cosm_const);

        df_dt(sixth_der, dt, metric, &dmetric_bound_dt_ksum, &dmetric_bound_dt_k2, c_fields_sum, c_fields_k2, q_fields_sum, q_fields_k2);
//#pragma omp parallel for
        for (int i = 0; i < lattice_size; ++i) {
            c_fields_sum->pi[i]  = c_fields->pi[i]  + dt * 0.5 * c_fields_k2->pi[i];
            c_fields_sum->chi[i] = c_fields->chi[i] + dt * 0.5 * c_fields_k2->chi[i];

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields_sum[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt * 0.5 * q_fields_k2[which_q_field]->phi[k][l][i];
                        q_fields_sum[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i] + dt * 0.5 * q_fields_k2[which_q_field]->chi[k][l][i];
                        q_fields_sum[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt * 0.5 * q_fields_k2[which_q_field]->pi[k][l][i];
                    }
                }
            }

        }
        dmetric_bound_dt_ksum.alpha       = dmetric_bound.alpha        + dt * 0.5 * dmetric_bound_dt_k2.alpha;
        dmetric_bound_dt_ksum.psi         = dmetric_bound.psi          + dt * 0.5 * dmetric_bound_dt_k2.psi;


        set_K_psi_alpha_beta(first_der, second_der, diag, metric, &dmetric_bound_dt_ksum, c_fields_sum, q_fields_sum, cosm_const);

        df_dt(sixth_der, dt, metric, &dmetric_bound_dt_ksum, &dmetric_bound_dt_k3, c_fields_sum, c_fields_k3, q_fields_sum, q_fields_k3);
//#pragma omp parallel for
        for (int i = 0; i < lattice_size; ++i) {
            c_fields_sum->pi[i]  = c_fields->pi[i]  + dt * c_fields_k3->pi[i];
            c_fields_sum->chi[i] = c_fields->chi[i] + dt * c_fields_k3->chi[i];

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields_sum[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt * q_fields_k3[which_q_field]->phi[k][l][i];
                        q_fields_sum[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i] + dt * q_fields_k3[which_q_field]->chi[k][l][i];
                        q_fields_sum[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt * q_fields_k3[which_q_field]->pi[k][l][i];
                    }
                }
            }
        }
        dmetric_bound_dt_ksum.alpha       = dmetric_bound.alpha        + dt * dmetric_bound_dt_k3.alpha;
        dmetric_bound_dt_ksum.psi         = dmetric_bound.psi          + dt * dmetric_bound_dt_k3.psi;


        set_K_psi_alpha_beta(first_der, second_der, diag, metric, &dmetric_bound_dt_ksum, c_fields_sum, q_fields_sum, cosm_const);

        df_dt(sixth_der, dt, metric, &dmetric_bound_dt_ksum, &dmetric_bound_dt_k4, c_fields_sum, c_fields_k4, q_fields_sum, q_fields_k4);
//#pragma omp parallel for
        for (int i = 0; i < lattice_size; ++i) {
            c_fields->pi[i]  = c_fields->pi[i]  + dt * (c_fields_k1->pi[i]  + 2.0 * c_fields_k2->pi[i]  + 2.0 * c_fields_k3->pi[i]  + c_fields_k4->pi[i])  / 6.0;
            c_fields->chi[i] = c_fields->chi[i] + dt * (c_fields_k1->chi[i] + 2.0 * c_fields_k2->chi[i] + 2.0 * c_fields_k3->chi[i] + c_fields_k4->chi[i]) / 6.0;

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt * (q_fields_k1[which_q_field]->phi[k][l][i] + 2.0 * q_fields_k2[which_q_field]->phi[k][l][i] + 2.0 * q_fields_k3[which_q_field]->phi[k][l][i] + q_fields_k4[which_q_field]->phi[k][l][i]) / 6.0;
                        q_fields[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i] + dt * (q_fields_k1[which_q_field]->chi[k][l][i] + 2.0 * q_fields_k2[which_q_field]->chi[k][l][i] + 2.0 * q_fields_k3[which_q_field]->chi[k][l][i] + q_fields_k4[which_q_field]->chi[k][l][i]) / 6.0;
                        q_fields[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt * (q_fields_k1[which_q_field]->pi[k][l][i]  + 2.0 * q_fields_k2[which_q_field]->pi[k][l][i]  + 2.0 * q_fields_k3[which_q_field]->pi[k][l][i]  + q_fields_k4[which_q_field]->pi[k][l][i]) / 6.0;
                    }
                }
            }
        }
        dmetric_bound.alpha       = dmetric_bound.alpha       + dt * (dmetric_bound_dt_k1.alpha       + 2.0 * dmetric_bound_dt_k2.alpha       + 2.0 * dmetric_bound_dt_k3.alpha       + dmetric_bound_dt_k4.alpha) / 6.0;
        dmetric_bound.psi         = dmetric_bound.psi         + dt * (dmetric_bound_dt_k1.psi         + 2.0 * dmetric_bound_dt_k2.psi         + 2.0 * dmetric_bound_dt_k3.psi         + dmetric_bound_dt_k4.psi)   / 6.0;// +dmetric_bound_dt_k4.psi;


        set_K_psi_alpha_beta(first_der, second_der, diag, metric, &dmetric_bound, c_fields, q_fields, cosm_const);


        if (isnan(dt) || metric->alpha[0] < 0.0 || metric->alpha[0] > 1.1) {
            printf("\nbad\n");
            printf("\nalpha=%.10f,", metric->alpha[0]);
            //break;
        }
        alphat[n] = 0.0;
        double minH=0.0;
        int minHloc;
        for (int i = 1; i < lattice_size - 1; ++i) {
            double r, psi, psi_prime, K_hat, H, chi, pi;
            double rm, psim, psi_primem, K_hatm, Hm, chim, pim;
            

            r = r_x(x[i+1]);
            psi = metric->psi[i+1];
            psi_prime = first_deriv(i+1, metric->psi);
            K_hat = metric->K_hat[i+1];
            H = r*psi_prime + 0.5 * psi + 0.25 * r*r*r / (psi * psi * psi) * K_hat;

            rm = r_x(x[i]);
            psim = metric->psi[i];
            psi_primem = first_deriv(i, metric->psi);
            K_hatm = metric->K_hat[i];
            Hm = rm*psi_primem + 0.5 * psim + 0.25 * rm *rm*rm/ (psim * psim * psim) * K_hatm;

            if (pow(psi, 2.0) + 2.0 * r * psi * psi_prime < 0) {
                printf("\nrpsi2 not monotonic at i=%d,", i);
                break;
            }

            if (H < minH) {
                minH = H;
                minHloc = i;
            }
            if (H > 0 && Hm < 0) {
                printf("\nH is zero at i=%d and the mass is m=%.10f,", i, r*psi*psi/ 2.0);

                horizont[n] = metric->psi[i] * metric->psi[i] * r_x(x[i]);// i;// rmat[n][i];// metric->psi[i] * metric->psi[i] * r_x(x[i]);
                time[n] = tmat[n][i];// tmat[n][i];
                ++nhor;
                //nu = 0.0;
                break;
            }
        }

        alphat[n] = metric->alpha[0]; 
    }
    save_alpha(alphat);
    save_horizon(horizont);
    save_time(time);

    return horizont[10];//R_max;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves the bilinears */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_bi_linears(double cosm_const, Classical_fields* c_fields, Quantum_fields** q_fields, Metric_Fields* metric) {

    double phi_phi_arr[lattice_size];
    double chi_chi_arr[lattice_size];
    double pi_pi_arr[lattice_size];
    double chi_pi_arr[lattice_size];
    double del_phi_phi_arr[lattice_size];

    for (int i = 0; i < lattice_size; i++) {

        double r, r_l;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
        double psi, beta, alpha;
        int l_value;

        double x[lattice_size];
        make_points(x);
        r = r_x(x[i]);
        alpha = metric->alpha[i];
        beta  = metric->beta[i];
        psi   = metric->psi[i];

        phi_phi = 0.0;
        chi_chi = 0.0;
        pi_pi = 0.0;
        chi_pi = 0.0;
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;

        //note that these modes are actually modes of phi, where Phi = r^l phi
        //Phi = r^l phi
        //Pi  = r^l pi
        //Psi = lr^{l-1} u + r^l psi
            //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;
                    r_l = pow(r, l_value);


                    /* PHI MODE */
                    Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);


                    /* CHI MODE */
                    Chi_mode = pow(metric->psi[i], 2.0) * l_value * pow(r, l_value - 2) * q_fields[which_q_field]->phi[k][l][i] + r_l * q_fields[which_q_field]->chi[k][l][i];

                    /* PI MODE */
                    Pi_mode = r_l * q_fields[which_q_field]->pi[k][l][i];


                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi = pi_pi +  ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi = chi_pi +  ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * (__real__(Pi_mode * conj(Chi_mode)));
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq +  ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);



                }
            }
        }

        phi_phi_arr[i] = phi_phi;
        chi_chi_arr[i] = chi_chi * r * r;
        pi_pi_arr[i] = pi_pi;
        chi_pi_arr[i] = chi_pi;
        del_phi_phi_arr[i] = del_theta_phi_del_theta_phi_over_r_sq;
    }

    // saving stress-energy tensor for different time steps
    FILE* finout;
    finout = fopen("phiphi.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.100f ", phi_phi_arr[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("chichi.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout1, "%.100f ", chi_chi_arr[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("chipi.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout2, "%.100f ", chi_pi_arr[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("pipi.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout3, "%.100f ", pi_pi_arr[m]);
    }
    fclose(finout3);

    FILE* finout4;
    finout4 = fopen("phithetaphitheta.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout4, "%.100f ", del_phi_phi_arr[m]);
    }
    fclose(finout4);

}
void free_memory(Classical_fields* c_fields, Quantum_fields** q_fields) {
    free(c_fields->pi);
    free(c_fields->chi);
    free(c_fields);

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(q_fields[which_q_field]->phi[k][l]);
                free(q_fields[which_q_field]->chi[k][l]);
                free(q_fields[which_q_field]->pi[k][l]);
            }
            free(q_fields[which_q_field]->phi[k]);
            free(q_fields[which_q_field]->chi[k]);
            free(q_fields[which_q_field]->pi[k]);
        }
        free(q_fields[which_q_field]->phi);
        free(q_fields[which_q_field]->chi);
        free(q_fields[which_q_field]->pi);

        free(q_fields[which_q_field]);
    }
    free(q_fields);
}

/* Main */
void main() {


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* DEFINE VARIABLES AND ALLOCATE THEIR MEMORY*/
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Classical_fields* c_fields;
    Classical_fields* c_fields_k1;
    Classical_fields* c_fields_k2;
    Classical_fields* c_fields_k3;
    Classical_fields* c_fields_k4;
    Classical_fields* c_fields_sum;

    c_fields = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields->chi = (double*)malloc(lattice_size * sizeof(double));

    c_fields_k1 = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields_k1->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_k1->chi = (double*)malloc(lattice_size * sizeof(double));

    c_fields_k2 = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields_k2->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_k2->chi = (double*)malloc(lattice_size * sizeof(double));

    c_fields_k3 = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields_k3->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_k3->chi = (double*)malloc(lattice_size * sizeof(double));

    c_fields_k4 = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields_k4->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_k4->chi = (double*)malloc(lattice_size * sizeof(double));

    c_fields_sum = (Classical_fields*)malloc(sizeof(Classical_fields));
    c_fields_sum->pi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_sum->chi = (double*)malloc(lattice_size * sizeof(double));

    Metric_Fields* metric;

    metric = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric->alpha = (double*)malloc(lattice_size * sizeof(double));
    metric->psi = (double*)malloc(lattice_size * sizeof(double));
    metric->K_hat = (double*)malloc(lattice_size * sizeof(double));
    metric->beta = (double*)malloc(lattice_size * sizeof(double));

    Quantum_fields** q_fields;
    Quantum_fields** q_fields_k1;
    Quantum_fields** q_fields_k2;
    Quantum_fields** q_fields_k3;
    Quantum_fields** q_fields_k4;
    Quantum_fields** q_fields_sum;

    q_fields = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_k1 = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_k2 = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_k3 = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_k4 = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_sum = (Quantum_fields**)malloc(number_of_q_fields * sizeof(Quantum_fields*));

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        q_fields[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_k1[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_k2[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_k3[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_k4[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_sum[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));

        q_fields[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_k1[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k1[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k1[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_k2[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k2[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k2[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_k3[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k3[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k3[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_k4[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k4[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_k4[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_sum[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_sum[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_sum[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));


        for (int k = 0; k < number_of_k_modes; k++) {

            q_fields[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_k1[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k1[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k1[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_k2[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k2[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k2[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_k3[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k3[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k3[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_k4[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k4[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_k4[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_sum[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_sum[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_sum[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));


            for (int l = 0; l < number_of_l_modes; ++l) {

                q_fields[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_k1[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k1[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k1[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_k2[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k2[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k2[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_k3[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k3[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k3[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_k4[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k4[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_k4[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_sum[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_sum[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_sum[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size * sizeof(__complex__ double));

            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* DEFINE MATRICES AND ALLOCATE MEMORY FOR THEM*/
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    double** first_der;
    first_der = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        first_der[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    double** second_der;
    second_der = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        second_der[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    double** sixth_der;
    sixth_der = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        sixth_der[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    double** place_hold;
    place_hold = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        place_hold[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    double** diag;
    diag = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        diag[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* INITIALISE POINTS AND DEFINE MATRICES TO SAVE */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double x[lattice_size], r[lattice_size];
    make_points(x);
    save_points(x);
    set_der_mats(first_der, second_der, diag);

    //make_sixth_der_mat(first_der, sixth_der, place_hold);
    double** fieldmat;
    double** tmat;
    double** rmat;
    fieldmat = (double**)malloc(evolve_time_int / divid * sizeof(double*));
    for (int i = 0; i < evolve_time_int / divid; i++) {
        fieldmat[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    rmat = (double**)malloc(evolve_time_int / divid * sizeof(double*));
    for (int i = 0; i < evolve_time_int / divid; i++) {
        rmat[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    tmat = (double**)malloc(evolve_time_int / divid * sizeof(double*));
    for (int i = 0; i < evolve_time_int / divid; i++) {
        tmat[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double stressmax = 0.0;
    double cosm_const = 0.0;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ACTUAL INITIAL CONDITIONS, INITIALISATION AND EVOLUTION */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    set_zero(c_fields_k1, q_fields_k1);
    set_zero(c_fields_k2, q_fields_k2);
    set_zero(c_fields_k3, q_fields_k3);
    set_zero(c_fields_k4, q_fields_k4);
    set_zero(c_fields_sum, q_fields_sum);

    initial_conditions(c_fields, metric);

    initial_conditions_quantum(c_fields, q_fields, metric);

    initialise_quantum(c_fields, q_fields, metric);

    cosm_const = set_cosm_constant(c_fields, q_fields, metric);
    printf("The cosmological constant is %.10f,\n", cosm_const);


    initialise_metric(first_der, second_der, diag, metric, c_fields, q_fields, cosm_const);

    stressmax = full_evolution(first_der, second_der, sixth_der, diag, metric, c_fields, c_fields_k1, c_fields_k2, c_fields_k3, c_fields_k4, c_fields_sum,
                                 q_fields, q_fields_k1, q_fields_k2, q_fields_k3, q_fields_k4, q_fields_sum,
                                 cosm_const, fieldmat, rmat, tmat);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* SAVE SOME FIELDS */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    save_field_t(fieldmat);
    save_t_mat(tmat);
    save_r_mat(rmat);


    double rpoints[lattice_size];
    for (int i = 0; i < lattice_size; i++) {
        double r;

        r = r_x(x[i]);
        rpoints[i] = r;
    }

    save_bi_linears(cosm_const, c_fields, q_fields, metric);

    FILE* finout1;
    finout1 = fopen("horizon.txt", "w");
    for (int n = 0; n < reps; ++n) {
        fprintf(finout1, "%.200f ", stressmax);
    }
    fclose(finout1);

    save_rpoints(rpoints);
    save_alphar(metric->alpha);
    save_beta(metric->beta);
    save_Khat(metric->K_hat);
    save_psi(metric->psi);
    save_psi_prime(metric->psi);
    save_chi(c_fields->chi);
    save_pi(c_fields->pi);

    for (int i = 0; i < lattice_size; i++) {
        r[i] = r_x(x[i]);
    }
    save_points(r);



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* FREE ALL THE MEMORY */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    free_memory(c_fields, q_fields);
    free_memory(c_fields_k1, q_fields_k1);
    free_memory(c_fields_k2, q_fields_k2);
    free_memory(c_fields_k3, q_fields_k3);
    free_memory(c_fields_k4, q_fields_k4);
    free_memory(c_fields_sum, q_fields_sum);

    free(metric->alpha);
    free(metric->psi);
    free(metric->K_hat);
    free(metric->beta);
    free(metric);


    for (int i = 0; i < lattice_size; i++) {
        free(first_der[i]);
    }
    free(first_der);

    for (int i = 0; i < lattice_size; i++) {
        free(second_der[i]);
    }
    free(second_der);

    for (int i = 0; i < lattice_size; i++) {
        free(sixth_der[i]);
    }
    free(sixth_der);

    for (int i = 0; i < lattice_size; i++) {
        free(place_hold[i]);
    }
    free(place_hold);

    for (int i = 0; i < lattice_size; i++) {
        free(diag[i]);
    }
    free(diag);
    for (int i = 0; i < evolve_time_int / divid; i++) {
        free(fieldmat[i]);
    }
    free(fieldmat);
    for (int i = 0; i < evolve_time_int / divid; i++) {
        free(rmat[i]);
    }
    free(rmat);
    for (int i = 0; i < evolve_time_int / divid; i++) {
        free(tmat[i]);
    }
    free(tmat);

    printf("done");
}
