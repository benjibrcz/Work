///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
This code was written by Benjamin Berczi as part of the PhD project titled "Simulating Semiclassical Black Holes" from the University of Nottingham.

It is a self-contained C file that simulates a massless quantum scalar field coupled to Einstein gravity in the double null formulation.

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
#include <complex.h>
/* CONSANTS */
#define         PI                                 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define         M_P                                1.0                                                                                   //sqrt(8*PI)
#define         c                                  1.0                                                                                   //speed of light

/* GRID PARAMETERS */
#define         u_size                             513                                                                                   // size of grid in u direction
#define         v_size                             513                                                                                   // size of grid in v direction
#define         du                                 0.02                                                                                  // step size in u direction
#define         dv                                 0.02                                                                                  // step size in v direction
#define         u_max                              du*(u_size-1)
#define         v_max                              dv*(v_size-1)
#define         N_HALF                             4
#define         r0                                 0.0

/* SCALAR FIELD PARAMETERS */
#define         amplitude                          0.0                                                                                   // initial amplitude
#define         mass                               0.0                                                                                   // initial mass
#define         initial_width                      1.0                                                                                   // initial width of the gaussian scalar field
#define         initial_radius                     5.0                                                                                   // initial radius of the gaussian scalar field

/* QUANTUM OR CLASSICAL SIMULATION */
#define         hbar                               0                                                                                     // set to 1 for quantum, 0 for classical. This just sets the backreaction, and is in set_bi_linears.c, the quantum modes are still evolved
#define         coherent_state_switch              1                                                                                     // set to 0 to just have the mode functions

/* QUANTUM GHOST FIELD PARAMETERS */
#define         number_of_q_fields                 6                                                                                     // number of quantum fields, 1 real, 5 ghosts for regularisation
#define         muSq                               0.0                                                                                   // mass of scalar field
#define         mSqGhost                           1.0                                                                                   // base mass of the Pauli-Villars regulator fields
double          massSq[number_of_q_fields]         = { muSq, mSqGhost, 3.0 * mSqGhost, mSqGhost, 3.0 * mSqGhost, 4.0 * mSqGhost };       // masses of the ghost fields
double          ghost_or_physical[6]               = { 1 , -1 , 1 , -1 , 1, -1 };                                                        // distinguishing between the real and ghost fields

/* QUANTUM MODE PARAMETERS */
#define         k_min                              1.0*PI/15.0 //PI/6.0;//5.0*2.0*PI/(lattice_size*dr);                                  // minimum value of k, also =dk
#define         dk                                 k_min            
#define         number_of_k_modes                  1                                                                                     // number of k modes
#define         number_of_l_modes                  1                                                                                     // number of l modes
#define         k_start                            0
#define         l_start                            0                                                                                     //the range of l is l_start, l_start+l_step, l_start+2l_step...
#define         l_step                             1


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Creating a structure for the variables */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fields {
    double* phi;
    double* sigma;
    double* r;
    double* D_phi;
    double* D_sigma;
    double* K_phi;
    double* K_r;
    double* D_r;
    double* mu;
    double* lambda;
};
typedef struct fields Fields;

struct q_fields {
    __complex__ double*** phi;
    __complex__ double*** D_phi;
    __complex__ double*** K_phi;
    __complex__ double*** mu;
};
typedef struct q_fields Q_Fields;

struct initial_fields_u0 {
    double* D_phi_u0;
    double* D_sigma_u0;
};
typedef struct initial_fields Initial_Fields;

struct bi_linears {
    double* phi_phi;
    double* Dphi_Dphi;
    double* Kphi_Kphi;
    double* Dphi_Kphi;
    double* del_theta_phi_del_theta_phi;
};
typedef struct bi_linears Bi_Linears;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Making the points for the spatial grid*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void make_points(double N, double v[v_size]) {
    double v_min = (pow(2.0, N) - 1.0) / pow(2.0, N) * v_max;
    double dvv = dv / pow(2.0, N);  
    for (int i = 0; i < v_size; ++i) {
        v[i] = v_min + i * dvv;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions to save things */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_points(double v[v_size]) {
    FILE* pointsout;
    pointsout = fopen("vpoints2.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(pointsout, "%.20f ", v[m]);
    }
    fclose(pointsout);
}
void save_horizonv(double hor[u_size]) {
    FILE* pointsout;
    pointsout = fopen("horizonv.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(pointsout, "%.20f ", hor[m]);
    }
    fclose(pointsout);
}
void save_horizonu(double hor[u_size]) {
    FILE* pointsout;
    pointsout = fopen("horizonu.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(pointsout, "%.20f ", hor[m]);
    }
    fclose(pointsout);
}
void save_upoints() {
    FILE* pointsout;
    pointsout = fopen("upoints2.txt", "w");
    for (int m = 0; m < u_size; ++m) {
        fprintf(pointsout, "%.20f ", m*du);
    }
    fclose(pointsout);
}
void save_Dphiq(__complex__ double* field) {
    FILE* finout;
    finout = fopen("Dphiq.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.100f ", __real__ field[m]);
    }
    fclose(finout);
}
void save_Kphiq(__complex__ double* field) {
    FILE* finout;
    finout = fopen("Kphiq.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.100f ", __real__ field[m]);
    }
    fclose(finout);
}
void save_muphiq(__complex__ double* field) {
    FILE* finout;
    finout = fopen("muphiq.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.100f ", __real__ field[m]);
    }
    fclose(finout);
}
void save_phi(double* field) {
    FILE* finout;
    finout = fopen("phi_null2.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_Dphi(double* field) {
    FILE* finout;
    finout = fopen("Dphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_Kphi(double* field) {
    FILE* finout;
    finout = fopen("Kphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_muphi(double* field) {
    FILE* finout;
    finout = fopen("mu.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_r1(double* field) {
    FILE* finout;
    finout = fopen("r_null2.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_sigma(double* field) {
    FILE* finout;
    finout = fopen("sigma_null2.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_constr(double constr[v_size]) {
    FILE* finout;
    finout = fopen("constr_null2.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", constr[m]);
    }
    fclose(finout);
}
void save_r(double** field) {

    FILE* finout;
    finout = fopen("phiphi_matrix100.txt", "w");
    for (int n = 0; n < N_HALF *(u_size - 1) / 2; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < v_size; ++m) {
            fprintf(finout, "%.100f ", field[n][m]);
        }
    }
    fclose(finout);
}
void save_phi0(double* field) {
    FILE* finout;
    finout = fopen("phi01.txt", "w");
    for (int n = 0; n < N_HALF * (u_size - 1) / 2; n++) {
        fprintf(finout, "%.20f ", field[n]);
    }
    fclose(finout);
}
void save_vmat() {
    double v[v_size];
    FILE* finout;
    finout = fopen("v_matrix2.txt", "w");
    for (int n_halved = 0; n_halved < N_HALF; n_halved++) {
        make_points(n_halved, v);
        for (int n = 0; n < (u_size - 1) / 2; n++) {
            fprintf(finout, "\n");
            for (int m = 0; m < v_size; ++m) {
                fprintf(finout, "%.100f ", v[m]);
            }
        }
    }
    fclose(finout);
}
void save_u() {
    //double u[N_HALF * (u_size - 1) / 2];
    double T = -dv;
    //u[0] = 0.0;
    FILE* finout;
    finout = fopen("u_points.txt", "w");
    for (int n_halved = 0; n_halved < N_HALF; n_halved++) {
        for (int n = 0; n < (u_size - 1) / 2; n++) {
            fprintf(finout, "\n");
            T = T + dv / pow(2.0, n_halved);
            for (int m = 0; m < v_size; ++m) {
                fprintf(finout, "%.20f ", T);
            }

        }
    }
    fclose(finout);
}
void save_u0(double* u) {
    double T = 0.0;
    FILE* finout;
    finout = fopen("u0.txt", "w");
    for (int n = 0; n < N_HALF * (u_size - 1) / 2; n++) {
        fprintf(finout, "%.20f ", u[n]);

    }
    fclose(finout);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Derivative function for real fields */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv_vo2(double dvv, double x1,double x2,double x3,double x4) {
    double der = 0.0;
    return der = (x2 - x1 + x4  - x3) / (2.0 * dvv);
}
double first_deriv_uo2(double dvv, double x1, double x2, double x3, double x4) {
    double der = 0.0;
    return der = (x3 - x1 + x4 - x2) / (2.0 * dvv);
}
double first_deriv_vo1(double dvv, double x1, double x2) {
    double der = 0.0;
    return der = (x2 - x1) / (1.0*dvv);
}
double first_deriv_uo1(double dvv, double x1, double x3) {
    double der = 0.0;
    return der = (x3 - x1) / (1.0*dvv);
}
double first_derivv(double dvv, int m, double* field) {
    double der = 0.0;


    if (m == 0) {
        return der = 0.0;
    }
    if (m == 1) {
        return der = 0.0;// (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[1]) / (12.0 * dvv);
    }

    if (m == v_size - 1) {
        return der = (3.0 * field[m - 4] - 16.0 * field[m - 3] + 36.0 * field[m - 2] - 48.0 * field[m - 1] + 25.0 * field[m]) / (12 * dvv);
    }
    if (m == v_size - 2) {
        return der = (-field[m - 3] + 6.0 * field[m - 2] - 18.0 * field[m - 1] + 10.0 * field[m] + 3.0 * field[m + 1]) / (12 * dvv);
    }
    if (2 <= m < v_size - 2) {

        return der = (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 2]) / (12.0 * dvv);

    }
}
double first_derivu(double dvv, int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m + 1] - field[m]) / (dvv);
    }
    if (m == u_size - 1) {
        return der = (field[m] - field[m - 1]) / (dvv);
    }
    if (m != 0 && m != u_size - 1) {
        return der = (field[m + 1] - field[m - 1]) / (2.0 * dvv);
    }
}
double second_derivv(double dvv, int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m+1] + field[m +2]) / (dvv * dvv);
    }
    if (m == v_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (dvv * dvv);
    }
    if (m != 0 && m != v_size - 1 ) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (dvv * dvv);
    }
}
double second_derivu(double dvv, int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m+1] + field[m +2]) / (dvv * dvv);
    }
    if (m == u_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (dvv * dvv);
    }
    if (m != 0 && m != u_size - 1 ) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (dvv * dvv);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* First derivative function for complex fields, uses 20 neighbouring points at the moment*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double sixth_deriv_comp(int i, int n, double dvv,  __complex__ double* field) {
    __complex__ double der = 0.0;

    if (3 < i < v_size - 3) {
        der = (field[i + 3] - 6.0 * field[i + 2] + 15.0 * field[i+1] - 20.0 * field[i] + 15.0 * field[i - 1] - 6.0 * field[i - 2] + field[i - 3]) / pow(dvv, 6.0);


    }
    if (i < n+3) {

        der = (field[i] - 6.0 * field[i + 1] + 15.0 * field[i + 2] - 20.0 * field[i + 3] + 15.0*field[i + 4] - 6.0*field[i+5] + 1.0*field[i+6]) / pow(dvv, 6.0);
    }
    if (i>v_size-4) {
        der = 0.0;
       
    }
    return der;
}
__complex__ double fourth_deriv_comp(int i, int n, double dvv, __complex__ double* field) {
    __complex__ double der = 0.0;

    if (2 < i < v_size - 2) {
        der = (field[i + 2] - 4.0 * field[i + 1] + 6.0 * field[i] - 4.0 * field[i - 1] + field[i - 2]) / pow(1.0 * dvv, 4.0);


    }
    if (i < n + 2) {

        der = (field[i] - 4.0 * field[i + 1] + 6.0 * field[i + 2] - 4.0 * field[i + 3] + field[i + 4]) / pow(dvv, 4.0);
    }
    if (i > v_size - 3) {
        der = 0.0;
        
    }
    return der;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the classical variables */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void charac_initial_conditions(Fields* fields) {
    double v[v_size];
    make_points(0, v);

    /* initial v slice */
    for (int i = 0; i < v_size; ++i) {
        fields->D_sigma[i] = 0.0;
        fields->phi[i] = amplitude * v[i] * v[i] * exp(-1.0 / 2.0 * pow((v[i] - initial_radius) / initial_width, 2.0));
        fields->D_phi[i] = -((v[i] - initial_radius) / (initial_width * initial_width)) * fields->phi[i] + 2.0 * amplitude * v[i] * exp(-1.0 / 2.0 * pow((v[i] - initial_radius) / initial_width, 2.0));

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
/* These functions provides the initial profile functions */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double phi_mode_profile_0(double k, int l, double vv) {
    double v = vv / 2.0;
    return (sqrt(k / PI) * cexp(-I*k*v)*gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l));
}
__complex__ double phi_mode_profile_0_prime(double k, int l, double vv) {
    double v = vv / 2.0;
    return 0.5*( (-k * sqrt(k / PI) * cexp(-I * k * v) * gsl_sf_bessel_jl_safe(l + 1, k * v) / pow(v, l)) - I *k* (sqrt(k / PI) * cexp(-I * k * v) * gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l)));
}
__complex__ double phi_mode_profile_0_dot(double k, int l, double vv) {
    double v = vv / 2.0;
    return 0.5 * ((k * sqrt(k / PI)* cexp(-I * k * v) * gsl_sf_bessel_jl_safe(l + 1, k * v) / pow(v, l)) - I * k * (sqrt(k / PI) * cexp(-I * k * v) * gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l)));
}
__complex__ double phi_mode_profile_massive(double msq, double k, int l, double vv) {
    double v = vv / 2.0;
    return (k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I *sqrt(k * k + msq) * v) * gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l));
}
__complex__ double phi_mode_profile_massive_prime(double msq, double k, int l, double vv) {
    double v = vv / 2.0;
    return 0.5 * ((-k * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * v) * gsl_sf_bessel_jl_safe(l + 1, k * v) / pow(v, l)) - I *sqrt(k * k + msq) * (k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * v) * gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l)));
}
__complex__ double phi_mode_profile_massive_dot(double msq, double k, int l, double vv) {
    double v = vv / 2.0;
    return 0.5 * ((k * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * v) * gsl_sf_bessel_jl_safe(l + 1, k * v) / pow(v, l)) - I * sqrt(k * k + msq) * (k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * v) * gsl_sf_bessel_jl_safe(l, k * v) / pow(v, l)));
}
__complex__ double phi_mode_profile_0_u(double k, int l, double vv, double uu) {
    double r = (vv-uu) / 2.0;
    double t = (vv+uu) / 2.0;
    return sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l) ;
}
__complex__ double phi_mode_profile_0_prime_u(double k, int l, double vv, double uu) {
    double v = vv / 2.0;
    double r = (vv - uu) / 2.0;
    double t = (vv + uu) / 2.0;
    return 0.5 * ( -k * sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l+1, k * r) / pow(r, l) - I * k * sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l) );
}
__complex__ double phi_mode_profile_0_dot_u(double k, int l, double vv, double uu) {
    double v = vv / 2.0;
    double r = (vv - uu) / 2.0;
    double t = (vv + uu) / 2.0;
    return 0.5 * ( k * sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l)  - I * k * sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l) );
}
__complex__ double phi_mode_profile_massive_u(double msq, double k, int l, double vv, double uu) {
    double v = vv / 2.0;
    double r = (vv - uu) / 2.0;
    double t = (vv + uu) / 2.0;
    return k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l);
}
__complex__ double phi_mode_profile_massive_prime_u(double msq, double k, int l, double vv, double uu) {
    double v = vv / 2.0;
    double r = (vv - uu) / 2.0;
    double t = (vv + uu) / 2.0;
    return 0.5 * ( -k * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l)  - I * sqrt(k * k + msq) * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l));
}
__complex__ double phi_mode_profile_massive_dot_u(double msq, double k, int l, double vv, double uu) {
    double v = vv / 2.0;
    double r = (vv - uu) / 2.0;
    double t = (vv + uu) / 2.0;
    return 0.5 * ( k * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l) - I * sqrt(k * k + msq) * k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l) );
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initial conditions for the quantum fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void charac_initial_conditions_quantum(Q_Fields** q_fields) {
    double v[v_size];
    make_points(0, v);
    double k_wavenumber, omega_phi;
    int l_value;
    /* initial v slice */
    for (int k = 0; k < number_of_k_modes; ++k) {
        k_wavenumber = (k_start + (k + 1)) * k_min;
        for (int l = 0; l < number_of_l_modes; ++l) {
            l_value = l_start + l * l_step;
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                omega_phi = sqrt(k_wavenumber * k_wavenumber + massSq[which_q_field]);
                if (massSq[which_q_field] == 0) {
                    for (int i = 1; i < v_size; ++i) {
                        q_fields[which_q_field]->phi[k][l][i]   = phi_mode_profile_0      (k_wavenumber, l_value, v[i]);
                        q_fields[which_q_field]->D_phi[k][l][i] = phi_mode_profile_0_prime(k_wavenumber, l_value, v[i]);
                        q_fields[which_q_field]->K_phi[k][l][i] = phi_mode_profile_0_dot(k_wavenumber, l_value, v[i]);
                    }
                }
                else {
                    for (int i = 1; i < v_size; ++i) {
                        q_fields[which_q_field]->phi[k][l][i]   = phi_mode_profile_massive      (massSq[which_q_field], k_wavenumber, l_value, v[i]);
                        q_fields[which_q_field]->D_phi[k][l][i] = phi_mode_profile_massive_prime(massSq[which_q_field], k_wavenumber, l_value, v[i]);
                        q_fields[which_q_field]->K_phi[k][l][i] = phi_mode_profile_massive_dot  (massSq[which_q_field], k_wavenumber, l_value, v[i]);
                    }
                }
                q_fields[which_q_field]->phi[k][l][0]   = k_wavenumber / sqrt(PI * omega_phi) * pow(k_wavenumber, l_value) / gsl_sf_doublefact(2 * l_value + 1);
                q_fields[which_q_field]->D_phi[k][l][0] = -I * 0.5 * omega_phi * q_fields[which_q_field]->phi[k][l][0];
                q_fields[which_q_field]->K_phi[k][l][0] = -I * 0.5 * omega_phi * q_fields[which_q_field]->phi[k][l][0];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Initial conditions for the functions at the centre */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_initial_conds_v0(int n, Fields* fields){
    fields->sigma[n] = fields->sigma[n + 1];
    fields->r[n]     = 0.0;
    fields->D_r[n]   = 0.5 * exp(fields->sigma[n] / 2.0);
    fields->phi[n]   = fields->phi[n + 1];
    fields->K_r[n]   = -0.5 * exp(fields->sigma[n] / 2.0);
    fields->K_phi[n] = fields->D_phi[n];
}
void set_initial_conds_v0_quantum(int n, Q_Fields** q_fields, Q_Fields** q_fields_old, Q_Fields** q_fields_old_old) {
    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                q_fields[which_q_field]->phi[k][l][n]     = q_fields[which_q_field]->phi[k][l][n+1];
                q_fields[which_q_field]->K_phi[k][l][n]   = q_fields[which_q_field]->D_phi[k][l][n];
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Set Minkowski spacetime values for the initial fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_minkowski(Fields* fields) {

    for (int i = 0; i < v_size; ++i) {
        fields->r[i]      =  0.5*i*dv;
        fields->D_r[i]    =  0.5;
        fields->K_r[i]    = -0.5;
        fields->sigma[i]  =  0.0;
        fields->mu[i]     =  0.0;
        fields->lambda[i] =  0.0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions to find the v derivative of fields to find them at each time step */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_sigma(int i, int n, double dvv, Fields* fields) {
    double sigma_prime;

    sigma_prime = fields->D_sigma[i];

    fields->sigma[i + 1] = fields->sigma[i] + dvv * sigma_prime;

}
void find_r_D_r(int i, int n, double dvv, Fields* fields, Bi_Linears* bi_linear) {
    double r_prime, D_r_prime;

    r_prime   = fields->D_r[i];
    D_r_prime = fields->D_r[i] * fields->D_sigma[i] -0.5 * fields->r[i] * bi_linear->Dphi_Dphi[i];

    fields->r[i + 1]   = fields->r[i] + dvv * r_prime;
    fields->D_r[i + 1] = fields->D_r[i] + dvv * D_r_prime;

}
void find_phi(int i, int n, double dvv, Fields* fields) {
    double phi_prime;

    phi_prime = fields->D_phi[i];

    fields->phi[i + 1] = fields->phi[i] + dvv * phi_prime;

}
void find_K_r(int i, int n, double dvv, Fields* fields) {
    double K_r_prime;

    K_r_prime = -(i != n ? fields->lambda[i] / fields->r[i] : 0.0);

    fields->K_r[i + 1] = fields->K_r[i] + dvv * K_r_prime;

}
void find_K_phi(int i, int n, double dvv, Fields* fields) {
    double K_phi_prime;

    fields->mu[i] = fields->D_r[i] * fields->K_phi[i] + fields->D_phi[i] * fields->K_r[i];

    K_phi_prime = -(i != n ? fields->mu[i] / fields->r[i] : 0.0);

    fields->K_phi[i + 1] = fields->K_phi[i] + dvv * K_phi_prime;

}
void find_q_phi(int i, int n, double dvv, Q_Fields** q_fields, Q_Fields** q_fields_old) {

    __complex__ double q_field_prime=0.0;

    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                if (i == n) {
                    q_field_prime = 0.5 * (q_fields[which_q_field]->D_phi[k][l][i] + q_fields[which_q_field]->D_phi[k][l][i + 1]);
                    q_fields[which_q_field]->phi[k][l][i + 1] = q_fields[which_q_field]->phi[k][l][i] + dvv * (q_field_prime - 0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->phi[k][l]));
                }
                if (i == n + 1) {
                    q_field_prime = 0.5 * (q_fields[which_q_field]->D_phi[k][l][i] + q_fields[which_q_field]->D_phi[k][l][i + 1]);
                    q_fields[which_q_field]->phi[k][l][i + 1] = q_fields[which_q_field]->phi[k][l][i] + dvv * (q_field_prime - 0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->phi[k][l]));

                }
                if (v_size-1 > i > n + 1) {
                    q_field_prime = 1.0/24.0 * (9.0*q_fields[which_q_field]->D_phi[k][l][i+1] + 19.0*q_fields[which_q_field]->D_phi[k][l][i] - 5.0 * q_fields[which_q_field]->D_phi[k][l][i-1] + 1.0 * q_fields[which_q_field]->D_phi[k][l][i-2]);
                    q_fields[which_q_field]->phi[k][l][i + 1] = q_fields[which_q_field]->phi[k][l][i] + dvv * (q_field_prime - 0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->phi[k][l]));
                }
                if (i == v_size - 1) {
                    q_field_prime = q_fields[which_q_field]->D_phi[k][l][i];
                    q_fields[which_q_field]->phi[k][l][i + 1] = q_fields[which_q_field]->phi[k][l][i] + dvv * (q_field_prime - 0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->phi[k][l]));

                }
            }
        }
    }
}
void find_q_K_phi(int i, int n, double dvv, Fields* fields, Q_Fields** q_fields, Q_Fields** q_fields_old, Bi_Linears* bi_linears) {


    fields->lambda[i] = fields->D_r[i] * fields->K_r[i] + 1.0 / 4.0 * exp(fields->sigma[i]) * (1.0 -bi_linears->del_theta_phi_del_theta_phi[i]);

    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                double k_wavenumber = (k_start + (k + 1)) * k_min;
                double l_value = l_start + l * l_step;

                __complex__ double q_field_prime, q_field_primef, phi, Kphi, Dphi, mu, K_f_prime_n, F_f;
                __complex__ double fk1, fk2, fk3, fk4;
                __complex__ double k1, k2, k3, k4;
                double r;

                fields->lambda[i + 1] = fields->D_r[i + 1] * fields->K_r[i + 1] + 1.0 / 4.0 * exp(fields->sigma[i + 1]) * (1.0 - bi_linears->del_theta_phi_del_theta_phi[i + 1]);
                
                if (i < v_size -1 ) {
                    fields->lambda[i + 1] = fields->D_r[i + 1] * fields->K_r[i + 1] + 1.0 / 4.0 * exp(fields->sigma[i + 1]) * (1.0 - bi_linears->del_theta_phi_del_theta_phi[i + 1]);

                    q_fields[which_q_field]->mu[k][l][i] = (i != n ? fields->D_r[i] * q_fields[which_q_field]->K_phi[k][l][i] + fields->K_r[i] * q_fields[which_q_field]->D_phi[k][l][i] : 0.0);


                    K_f_prime_n = (i != n ? -(l_value + 1.0) / fields->r[i] * q_fields[which_q_field]->mu[k][l][i]

                        - pow(l_value / fields->r[i], 2.0) * fields->lambda[i] * q_fields[which_q_field]->phi[k][l][i] : 0.0)

                        - 0.25 * massSq[which_q_field] * exp(fields->sigma[i]) * q_fields[which_q_field]->phi[k][l][i];
                        
                    F_f = pow(l_value / fields->r[i + 1], 2.0) * fields->lambda[i + 1] * q_fields[which_q_field]->phi[k][l][i + 1] + 0.25 * massSq[which_q_field] * q_fields[which_q_field]->phi[k][l][i + 1];

                    q_fields[which_q_field]->K_phi[k][l][i + 1] = (q_fields[which_q_field]->K_phi[k][l][i] + dvv * (-fields->K_r[i + 1] * (l_value + 1.0) / fields->r[i + 1] * q_fields[which_q_field]->D_phi[k][l][i + 1] - F_f - 0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->K_phi[k][l]))) / (1.0 + dvv * fields->D_r[i + 1] * (l_value + 1.0) / fields->r[i + 1]);
                    
                    
                }
                else {
                    q_fields[which_q_field]->mu[k][l][i] = fields->D_r[i] * q_fields[which_q_field]->K_phi[k][l][i] + fields->K_r[i] * q_fields[which_q_field]->D_phi[k][l][i];

                    q_field_prime = (i != n ? -(l_value + 1.0) / fields->r[i] * q_fields[which_q_field]->mu[k][l][i]

                        - pow(l_value / fields->r[i], 2.0) * fields->lambda[i] * q_fields[which_q_field]->phi[k][l][i] : 0.0)

                        - 0.25 * massSq[which_q_field] * exp(fields->sigma[i]) * q_fields[which_q_field]->phi[k][l][i];

                    q_fields[which_q_field]->K_phi[k][l][i + 1] = q_fields[which_q_field]->K_phi[k][l][i] + dvv * (q_field_prime -0.0 * pow(dvv, 3.0) * fourth_deriv_comp(i, n, dvv, q_fields_old[which_q_field]->K_phi[k][l]));
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
/* Function that calculates the bilinears of the matter field */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears(double cosm_const, int i, int n, double dvv, Bi_Linears* bi_linears, Fields* fields, Q_Fields** q_fields) {
    double r, r_l, rprime, rdot;
    double phi_phi, chi_chi, chi_pi, pi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    _Complex double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    int l_value;
    double v[v_size];
    make_points(0, v);
    

    phi_phi = 0.0;
    chi_chi = 0.0;
    chi_pi = 0.0;
    pi_pi = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if (coherent_state_switch != 0) {
        phi_phi   = fields->phi[i]   * fields->phi[i];
        chi_chi   = fields->D_phi[i] * fields->D_phi[i];
        pi_pi     = fields->K_phi[i] * fields->K_phi[i];
        chi_pi    = fields->D_phi[i] * fields->K_phi[i];
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    }

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = lr^{l-1} dr/du u + r^l pi
    //Psi = lr^{l-1} dr/dv u + r^l psi
    if (hbar != 0) {
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;

                    r      = fields->r[i];
                    r_l    = pow(r, l_value);
                    rprime = fields->D_r[i];
                    rdot   = fields->K_r[i];


                    /* PHI MODE */
                    Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);

                    if (i == n) {
                        Phi_mode_plus = pow(fields->r[i+1], l_value) * (q_fields[which_q_field]->phi[k][l][i + 1]);
                    }

                    /* CHI MODE */
                    if (l_value == 0) {
                        Chi_mode = q_fields[which_q_field]->D_phi[k][l][i];
                    }
                    else if (l_value == 1) {
                        Chi_mode = rprime * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->D_phi[k][l][i];
                    }
                    else {
                        Chi_mode = l_value * pow(r, l_value - 1) * rprime * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->D_phi[k][l][i]);
                    }

                    /* PI MODE */
                    if (l_value == 0) {
                        Pi_mode = q_fields[which_q_field]->K_phi[k][l][i];
                    }
                    else if (l_value == 1) {
                        Pi_mode = rdot * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->K_phi[k][l][i];
                    }
                    else {
                        Pi_mode = l_value * pow(r, l_value - 1) * rdot * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->K_phi[k][l][i]);
                    }

                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi   = pi_pi   + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi  = chi_pi  + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * 0.5*((__real__(Pi_mode * conj(Chi_mode))) + (__real__(Chi_mode * conj(Pi_mode))));

                    if (i != n) {
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);
                    }
                    else {//use the data at r=dr to estimate the r=0 case. This is only relevant for l=1
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode_plus) / (fields->r[i+1] * fields->r[i+1]);
                    }


                }
            }
        }
    }
    //printf("\n %.100f, ", norm(chi_mode));
    bi_linears->phi_phi[i] = phi_phi -2* cosm_const;
    bi_linears->Dphi_Dphi[i] = chi_chi;
    bi_linears->Kphi_Kphi[i] = pi_pi;
    bi_linears->Dphi_Kphi[i] = chi_pi + cosm_const/2.0;
    bi_linears->del_theta_phi_del_theta_phi[i] = del_theta_phi_del_theta_phi_over_r_sq - cosm_const;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the cosmological constant */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double set_cosm_constant(Fields* fields, Q_Fields** q_fields, Bi_Linears* bi_linears) {
    double rho, S_A, A, B;
    int i, n;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;

    A = 1.0;
    B = 1.0;

    i = 0;//buff_size+0;
    n = 0;
    double cosm_const = 0.0;
    double r, r_l, rprime, rdot;
    //double phi_phi, chi_chi, chi_pi, pi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    _Complex double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    int l_value;
    double v[v_size];
    make_points(0, v);


    phi_phi = 0.0;
    chi_chi = 0.0;
    chi_pi = 0.0;
    pi_pi = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if (coherent_state_switch != 0) {
        phi_phi = fields->phi[i] * fields->phi[i];
        chi_chi = fields->D_phi[i] * fields->D_phi[i];
        pi_pi = fields->K_phi[i] * fields->K_phi[i];
        chi_pi = fields->D_phi[i] * fields->K_phi[i];
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    }

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = lr^{l-1} dr/du u + r^l pi
    //Psi = lr^{l-1} dr/dv u + r^l psi
    //#pragma omp parallel for
    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                l_value = l_start + l * l_step;

                r = fields->r[i];
                r_l = pow(r, l_value);
                rprime = fields->D_r[i];
                rdot = fields->K_r[i];


                /* PHI MODE */
                Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);

                if (i == n) {
                    Phi_mode_plus = pow(fields->r[i + 1], l_value) * (q_fields[which_q_field]->phi[k][l][i + 1]);
                }

                /* CHI MODE */
                if (l_value == 0) {
                    Chi_mode = q_fields[which_q_field]->D_phi[k][l][i];
                }
                else if (l_value == 1) {
                    Chi_mode = rprime * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->D_phi[k][l][i];
                }
                else {
                    Chi_mode = l_value * pow(r, l_value - 1) * rprime * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->D_phi[k][l][i]);
                }

                /* PI MODE */
                if (l_value == 0) {
                    Pi_mode = q_fields[which_q_field]->K_phi[k][l][i];
                }
                else if (l_value == 1) {
                    Pi_mode = rdot * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->K_phi[k][l][i];
                }
                else {
                    Pi_mode = l_value * pow(r, l_value - 1) * rdot * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->K_phi[k][l][i]);
                }

                /* ACTUAL BILINEARS */
                phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                chi_chi = chi_chi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                pi_pi = pi_pi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                chi_pi = chi_pi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * 0.5 * ((__real__(Pi_mode * conj(Chi_mode))) + (__real__(Chi_mode * conj(Pi_mode))));

                if (i != n) {
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);
                }
                else {//use the data at r=dr to estimate the r=0 case. This is only relevant for l=1
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode_plus) / (fields->r[i + 1] * fields->r[i + 1]);
                }


            }
        }
    }

    return del_theta_phi_del_theta_phi_over_r_sq; //-chi_pi * 2.0;// 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Calculates initial configuration of the fields that have to be integrated in the v direction */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void v_evolution_initial(double cosm_const, Fields* fields, Q_Fields** q_fields, Bi_Linears* bi_linears) {
    double dvv = dv;

    // integrate out first order radial equations (sigma, r, D_r, phi, K_r, K_phi)


    for (int i = 0; i < v_size; ++i) {

        bi_linears->phi_phi[i] = fields->phi[i] * fields->phi[i];
        bi_linears->Dphi_Dphi[i] = fields->D_phi[i] * fields->D_phi[i];
        bi_linears->Kphi_Kphi[i] = fields->K_phi[i] * fields->K_phi[i];
        bi_linears->Dphi_Kphi[i] = fields->D_phi[i] * fields->K_phi[i];
        bi_linears->del_theta_phi_del_theta_phi[i] = 0.0;

        ///////////////////////////////////////////////////////////////////


        find_r_D_r(i, 0, dvv, fields, bi_linears);

        find_K_r(i, 0, dvv, fields);

        fields->lambda[i] = fields->D_r[i] * fields->K_r[i] + 1.0 / 4.0 * exp(fields->sigma[i]) * (1.0 - bi_linears->del_theta_phi_del_theta_phi[i]);


        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {


                    q_fields[which_q_field]->mu[k][l][i] = fields->D_r[i] * q_fields[which_q_field]->K_phi[k][l][i] + fields->K_r[i] * q_fields[which_q_field]->D_phi[k][l][i];

                    
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the fields that have to be integrated in the v direction at each time (u) step */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void v_evolution(double cosm_const, int n, int n_halved, Fields* fields, Q_Fields** q_fields, Q_Fields** q_fields_old, Q_Fields** q_fields_old_old, Bi_Linears* bi_linears) {

    double dvv = dv / pow(2.0, n_halved);

    // integrate out first order radial equations (sigma, r, D_r, phi, K_r, K_phi)
    set_initial_conds_v0        (n, fields);
    set_initial_conds_v0_quantum(n, q_fields, q_fields_old, q_fields_old_old);

    for (int i = n; i < v_size; ++i) {
        //////////////////////////////////////////////////////////////////////
        set_bi_linears(cosm_const, i, n, dvv, bi_linears, fields, q_fields); 
        //////////////////////////////////////////////////////////////////////

        find_sigma(i, n, dvv, fields);

        find_phi  (i, n, dvv, fields);

        find_q_phi(i, n, dvv, q_fields, q_fields_old);

        find_r_D_r(i, n, dvv, fields, bi_linears);

        find_K_r  (i, n, dvv, fields);

        find_K_phi(i, n, dvv, fields);

        find_q_K_phi(i, n, dvv, fields, q_fields, q_fields_old, bi_linears);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Finds the right hand side of the time derivative (u) of the function D_f */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double set_D_f_dot(int i, int k, int l, int which_q_field, int l_value, Fields* fields, Q_Fields** q_fields) {

    return (-(l_value + 1.0) / fields->r[i] * q_fields[which_q_field]->mu[k][l][i]

        - pow(l_value / fields->r[i], 2.0) * fields->lambda[i] * q_fields[which_q_field]->phi[k][l][i]

        - 0.25 * massSq[which_q_field] * exp(fields->sigma[i]) * q_fields[which_q_field]->phi[k][l][i]);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Evolution of the fields that must be integrated in the u direction */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void u_evolution(double cosm_const, int n, double n_halved, Fields* fields, Q_Fields** q_fields, Bi_Linears* bi_linears,
    Fields* fields_n, Q_Fields** q_fields_n, Bi_Linears* bi_linears_n,
    Fields* fields_nm1, Q_Fields** q_fields_nm1, Bi_Linears* bi_linears_nm1,
    Fields* fields_nm2, Q_Fields** q_fields_nm2, Bi_Linears* bi_linears_nm2) {


    double dvv = dv / pow(2.0, n_halved);
    double duu = du / pow(2.0, n_halved);

    for (int i = n; i < v_size; ++i) {
        fields->D_phi[i] = fields->D_phi[i] + duu * (- fields->mu[i] / fields->r[i] - 0.25 * mass * mass * exp(fields->sigma[i]) * fields->phi[i]);

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    double k_wavenumber = (k_start + (k + 1)) * k_min;
                    double l_value = l_start + l * l_step;
                    
                    q_fields[which_q_field]->D_phi[k][l][i] = q_fields[which_q_field]->D_phi[k][l][i]

                        + duu * (-(l_value + 1.0) / fields->r[i] * q_fields[which_q_field]->mu[k][l][i]

                            - pow(l_value / fields->r[i], 2.0) * fields->lambda[i] * q_fields[which_q_field]->phi[k][l][i] 

                            - 0.25 * massSq[which_q_field] * exp(fields->sigma[i]) * q_fields[which_q_field]->phi[k][l][i]);
                    
                }

            }
        }


        fields->D_sigma[i] = fields->D_sigma[i] + duu * ( 2.0 * fields->lambda[i] / pow(fields->r[i], 2.0) - bi_linears->Dphi_Kphi[i]);

    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions that calculates contraints in the v and u direction */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void check_constrsv(double constr[v_size],Fields* fields) {
    
    for (int i = 0; i < v_size; i++) {
        constr[i] = second_derivv(dv, i, fields->r) - first_derivv(dv, i, fields->r) * first_derivv(dv, i, fields->sigma) + fields->r[i] * pow( first_derivv(dv, i, fields->phi) , 2.0);
    }
}
void check_constrsu(double constr[u_size],Fields* fields) {
    
    for (int i = 0; i < v_size; i++) {
        constr[i] = second_derivu(dv, i, fields->r) - first_derivu(dv, i, fields->r) * first_derivu(dv, i, fields->sigma) + fields->r[i] * pow( first_derivu(dv, i, fields->phi) , 2.0);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that sets the functions at the previous time step before they get evolved */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_fields_old(Fields* fields, Q_Fields** q_fields, Fields* fields_n, Q_Fields** q_fields_n) {

    for (int i = 0; i < v_size; ++i) {
        fields_n->phi[i] = fields->phi[i];
        fields_n->r[i] = fields->r[i];
        fields_n->sigma[i] = fields->sigma[i];
        fields_n->D_sigma[i] = fields->D_sigma[i];
        fields_n->D_phi[i] = fields->D_phi[i];
        fields_n->K_phi[i] = fields->K_phi[i];
        fields_n->K_r[i] = fields->K_r[i];
        fields_n->D_r[i] = fields->D_r[i];
        fields_n->mu[i] = fields->mu[i];
        fields_n->lambda[i] = fields->lambda[i];

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    q_fields_n[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i];
                    q_fields_n[which_q_field]->D_phi[k][l][i] = q_fields[which_q_field]->D_phi[k][l][i];
                    q_fields_n[which_q_field]->K_phi[k][l][i] = q_fields[which_q_field]->K_phi[k][l][i];
                    q_fields_n[which_q_field]->mu[k][l][i] = q_fields[which_q_field]->mu[k][l][i];
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that sets the bilinears on the previous time step */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears_old(Bi_Linears* bi_linears, Bi_Linears* bi_linears_n) {
    for (int i = 0; i < v_size; ++i) {
        bi_linears_n->phi_phi[i] = bi_linears->phi_phi[i];
        bi_linears_n->Dphi_Dphi[i] = bi_linears->Dphi_Dphi[i];
        bi_linears_n->Kphi_Kphi[i] = bi_linears->Kphi_Kphi[i];
        bi_linears_n->Dphi_Kphi[i] = bi_linears->Dphi_Kphi[i];
        bi_linears_n->del_theta_phi_del_theta_phi[i] = bi_linears->del_theta_phi_del_theta_phi[i];
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that interpolates fields, this happens when the grid looses half of its points, at which point the fields are reinterpolated so that they populate all the grid points */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void interpolate_fields(Fields* fields, Q_Fields** q_fields, Fields* fields_old, Q_Fields** q_fields_old) { 

    for (int i = (v_size - 1) / 2; i < v_size; i++) {
        fields->phi[2 * (i - (v_size - 1)/2)]       = fields_old->phi[i];
        fields->sigma[2 * (i - (v_size - 1)/2)]     = fields_old->sigma[i];
        fields->r[2 * (i - (v_size - 1)/2)]         = fields_old->r[i];
        fields->D_phi[2 * (i - (v_size - 1) / 2)]   = fields_old->D_phi[i];
        fields->D_sigma[2 * (i - (v_size - 1) / 2)] = fields_old->D_sigma[i];
        fields->D_r[2 * (i - (v_size - 1) / 2)]     = fields_old->D_r[i];
        fields->K_phi[2 * (i - (v_size - 1) / 2)]   = fields_old->K_phi[i];
        fields->K_r[2 * (i - (v_size - 1) / 2)]     = fields_old->K_r[i];
        fields->mu[2 * (i - (v_size - 1) / 2)]      = fields_old->mu[i];
        fields->lambda[2 * (i - (v_size - 1) / 2)]  = fields_old->lambda[i];

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    q_fields[which_q_field]->phi[k][l][2 * (i - (v_size - 1) / 2)]   = q_fields_old[which_q_field]->phi[k][l][i];
                    q_fields[which_q_field]->D_phi[k][l][2 * (i - (v_size - 1) / 2)] = q_fields_old[which_q_field]->D_phi[k][l][i];
                    q_fields[which_q_field]->K_phi[k][l][2 * (i - (v_size - 1) / 2)] = q_fields_old[which_q_field]->K_phi[k][l][i];
                    q_fields[which_q_field]->mu[k][l][2 * (i - (v_size - 1) / 2)]    = q_fields_old[which_q_field]->mu[k][l][i];
                }
            }
        }
    }
    for (int i = (v_size - 1) / 2; i < v_size-1; i++) {
        fields->phi[2* (i - (v_size - 1)/2) + 1]     = (fields_old->phi[i]     + fields_old->phi[i + 1]) / 2.0;
        fields->sigma[2* (i - (v_size - 1)/2) + 1]   = (fields_old->sigma[i]   + fields_old->sigma[i + 1]) / 2.0;
        fields->r[2* (i - (v_size - 1)/2) + 1]       = (fields_old->r[i]       + fields_old->r[i + 1]) / 2.0;
        fields->D_phi[2* (i - (v_size - 1)/2) + 1]   = (fields_old->D_phi[i]   + fields_old->D_phi[i + 1]) / 2.0;
        fields->D_sigma[2* (i - (v_size - 1)/2) + 1] = (fields_old->D_sigma[i] + fields_old->D_sigma[i + 1]) / 2.0;
        fields->D_r[2* (i - (v_size - 1)/2) + 1]     = (fields_old->D_r[i]     + fields_old->D_r[i + 1]) / 2.0;
        fields->K_phi[2* (i - (v_size - 1)/2) + 1]   = (fields_old->K_phi[i]   + fields_old->K_phi[i + 1]) / 2.0;
        fields->K_r[2* (i - (v_size - 1)/2) + 1]     = (fields_old->K_r[i]     + fields_old->K_r[i + 1]) / 2.0;
        fields->mu[2* (i - (v_size - 1)/2) + 1]      = (fields_old->mu[i]      + fields_old->mu[i + 1]) / 2.0;
        fields->lambda[2* (i - (v_size - 1)/2) + 1]  = (fields_old->lambda[i]  + fields_old->lambda[i + 1]) / 2.0;

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    q_fields[which_q_field]->phi[k][l][2 * (i - (v_size - 1) / 2) + 1]   = (q_fields_old[which_q_field]->phi[k][l][i]   + q_fields_old[which_q_field]->phi[k][l][i + 1]) / 2.0;
                    q_fields[which_q_field]->D_phi[k][l][2 * (i - (v_size - 1) / 2) + 1] = (q_fields_old[which_q_field]->D_phi[k][l][i] + q_fields_old[which_q_field]->D_phi[k][l][i + 1]) / 2.0;
                    q_fields[which_q_field]->K_phi[k][l][2 * (i - (v_size - 1) / 2) + 1] = (q_fields_old[which_q_field]->K_phi[k][l][i] + q_fields_old[which_q_field]->K_phi[k][l][i + 1]) / 2.0;
                    q_fields[which_q_field]->mu[k][l][2 * (i - (v_size - 1) / 2) + 1]    = (q_fields_old[which_q_field]->mu[k][l][i]    + q_fields_old[which_q_field]->mu[k][l][i + 1]) / 2.0;
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Full evolution including the u and v integrations */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void full_evolution(double cosm_const, double** full_r, Fields* fields, Fields* fields_n, Fields* fields_nm1, Fields* fields_nm2,
    Q_Fields** q_fields, Q_Fields** q_fields_n, Q_Fields** q_fields_nm1, Q_Fields** q_fields_nm2,
    Bi_Linears* bi_linears, Bi_Linears* bi_linears_n, Bi_Linears* bi_linears_nm1, Bi_Linears* bi_linears_nm2) {

    double stress_max = 0.0;
    double hormax=0.0;
    double horizonv[v_size];
    double horizonu[v_size];
    double phi0[N_HALF * (u_size - 1) / 2];
    double u0[N_HALF * (u_size - 1) / 2];
    double T = -dv;
    set_fields_old(fields, q_fields, fields_n, q_fields_n);
    set_fields_old(fields, q_fields, fields_nm1, q_fields_nm1);
    set_fields_old(fields, q_fields, fields_nm2, q_fields_nm2);
    for (int n_halved=0; n_halved < N_HALF; n_halved++) {
        printf("n_halved=%d\n", n_halved);
        double dvv = dv / pow(2.0, n_halved);
        double duu = du / pow(2.0, n_halved);
        for (int n = 0; n < (u_size-1)/2; ++n) {

            T = T + exp(fields->sigma[n] / 2.0) * dv / pow(2.0, n_halved);

            phi0[n_halved * (u_size - 1) / 2 + n] = fields->phi[n];
            u0  [n_halved * (u_size - 1) / 2 + n] = T;
            printf("n=%d\n", n);
            horizonv[n] = 0.0;
            horizonu[n] = 0.0;
            for (int i = n; i < v_size; ++i) {
                double r1, r2, r3, r4, r_vderiv;
                r1 = fields_n->r[i - 1];
                r2 = fields_n->r[i];
                r3 = fields->r[i - 1];
                r4 = fields->r[i];
                r_vderiv = fields->D_r[i];
                if (r_vderiv < 0.0) {

                    if (hormax < r4) {
                        hormax = r4;
                        printf("r' negative at r=%.10f\n", (r4) / 1.0);
                    }
                    horizonv[i] = (i - n) * dvv;
                    horizonu[i] = n * duu;
                    break;
                }
            }

            if (stress_max < exp(-fields->sigma[n]) * (pow(fields->K_phi[n], 2.0) + pow(fields->D_phi[n], 2.0))) {
                stress_max = exp(-fields->sigma[n]) * (pow(fields->K_phi[n], 2.0) + pow(fields->D_phi[n], 2.0));
            }

            
            
            
            for (int m = n; m < v_size; ++m) {
                //set_bi_linears_q(cosm_const, m, n, dvv, bi_linears, fields, q_fields);
                full_r[n_halved * (u_size - 1) / 2 + n][m] = fields->r[m];
                if (fabs(full_r[n_halved * (u_size - 1) / 2 + n][m]) > 10.0) {
                    full_r[n_halved * (u_size - 1) / 2 + n][m] = 0.0;
                }
            }
            printf("DphiDphi at r=0 is %.10f\n", full_r[n_halved * (u_size - 1) / 2 + n][v_size-1]);
            
            set_fields_old(fields_nm1, q_fields_nm1, fields_nm2, q_fields_nm2);
            set_fields_old(fields_n, q_fields_n, fields_nm1, q_fields_nm1);
            set_fields_old(fields, q_fields, fields_n, q_fields_n);
            set_bi_linears_old(bi_linears_nm1, bi_linears_nm2);
            set_bi_linears_old(bi_linears_n, bi_linears_nm1);
            set_bi_linears_old(bi_linears, bi_linears_n);

            u_evolution(cosm_const, n + 1, n_halved, fields, q_fields, bi_linears, fields_n, q_fields_n, bi_linears_n, fields_nm1, q_fields_nm1, bi_linears_nm1, fields_nm2, q_fields_nm2, bi_linears_nm2);

            set_initial_conds_v0(n+1, fields);
            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                        if (n == 0) {
                            q_fields[which_q_field]->phi[k][l][n + 1] = q_fields_n[which_q_field]->phi[k][l][n + 2];
                        }
                        else {
                            q_fields[which_q_field]->phi[k][l][n + 1] = (4.0 * q_fields_n[which_q_field]->phi[k][l][n + 2] - q_fields_nm1[which_q_field]->phi[k][l][n + 3]) / 3.0;
                        }
                        q_fields[which_q_field]->K_phi[k][l][n + 1] = q_fields[which_q_field]->D_phi[k][l][n + 1];
                    }
                }
            }

            for (int i = n + 1; i < v_size-1; ++i) {
                set_bi_linears(cosm_const, i, n + 1, dvv, bi_linears, fields, q_fields);

                find_sigma  (i, n + 1, dvv, fields);
                find_phi    (i, n + 1, dvv, fields);
                find_q_phi  (i, n + 1, dvv, q_fields, q_fields_n);
                find_r_D_r  (i, n + 1, dvv, fields, bi_linears);
                find_K_r    (i, n + 1, dvv, fields);
                find_K_phi  (i, n + 1, dvv, fields);
                find_q_K_phi(i, n + 1, dvv, fields, q_fields, q_fields_n, bi_linears);
            }

        }
       
        
        set_fields_old(fields, q_fields, fields_n, q_fields_n);
        interpolate_fields(fields, q_fields, fields_n, q_fields_n);
        set_fields_old(fields, q_fields, fields_n, q_fields_n);
        
        
    }
    
    save_horizonu(horizonu);
    save_horizonv(horizonv);

    printf("\nmax stress  = %.10f\n", hormax);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves the bilinears */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_bi_linears(int n, double cosm_const, Fields* fields, Q_Fields** q_fields, Bi_Linears* bi_linears) {

    
    for (int i = 0; i < v_size; i++) {

        //set_bi_linears(cosm_const, i, 1, dv, bi_linears, fields, q_fields);

        double r, r_l, rprime, rdot;
        double phi_phi, chi_chi, chi_pi, pi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        _Complex double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
        int l_value;
        double v[v_size];
        make_points(0, v);


        phi_phi = 0.0;
        chi_chi = 0.0;
        chi_pi = 0.0;
        pi_pi = 0.0;
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;

        if (coherent_state_switch != 0) {
            phi_phi = fields->phi[i] * fields->phi[i];
            chi_chi = fields->D_phi[i] * fields->D_phi[i];
            pi_pi = fields->K_phi[i] * fields->K_phi[i];
            chi_pi = fields->D_phi[i] * fields->K_phi[i];
            del_theta_phi_del_theta_phi_over_r_sq = 0.0;
        }

        //note that these modes are actually modes of phi, where Phi = r^l phi
        //Phi = r^l phi
        //Pi  = lr^{l-1} dr/du u + r^l pi
        //Psi = lr^{l-1} dr/dv u + r^l psi
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;

                    r = fields->r[i];
                    r_l = pow(r, l_value);
                    rprime = fields->D_r[i];
                    rdot = fields->K_r[i];


                    /* PHI MODE */
                    Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);

                    if (i == n) {
                        Phi_mode_plus = pow(fields->r[i + 1], l_value) * (q_fields[which_q_field]->phi[k][l][i + 1]);
                    }

                    /* CHI MODE */
                    if (l_value == 0) {
                        Chi_mode = q_fields[which_q_field]->D_phi[k][l][i];
                    }
                    else if (l_value == 1) {
                        Chi_mode = rprime * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->D_phi[k][l][i];
                    }
                    else {
                        Chi_mode = l_value * pow(r, l_value - 1) * rprime * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->D_phi[k][l][i]);
                    }

                    /* PI MODE */
                    if (l_value == 0) {
                        Pi_mode = q_fields[which_q_field]->K_phi[k][l][i];
                    }
                    else if (l_value == 1) {
                        Pi_mode = rdot * q_fields[which_q_field]->phi[k][l][i] + r * q_fields[which_q_field]->K_phi[k][l][i];
                    }
                    else {
                        Pi_mode = l_value * pow(r, l_value - 1) * rdot * q_fields[which_q_field]->phi[k][l][i] + r_l * (q_fields[which_q_field]->K_phi[k][l][i]);
                    }

                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi = pi_pi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi = chi_pi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * 0.5 * ((__real__(Pi_mode * conj(Chi_mode))) + (__real__(Chi_mode * conj(Pi_mode))));

                    if (i != n) {
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);
                    }
                    else {//use the data at r=dr to estimate the r=0 case. This is only relevant for l=1
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode_plus) / (fields->r[i + 1] * fields->r[i + 1]);
                    }


                }
            }
        }

        //printf("\n %.100f, ", norm(chi_mode));
        bi_linears->phi_phi[i] = phi_phi - 2 * cosm_const;
        bi_linears->Dphi_Dphi[i] = chi_chi;
        bi_linears->Kphi_Kphi[i] = pi_pi;
        bi_linears->Dphi_Kphi[i] = chi_pi + cosm_const / 2.0;
        bi_linears->del_theta_phi_del_theta_phi[i] = del_theta_phi_del_theta_phi_over_r_sq - cosm_const;

    }


    // save ham constr
    // saving stress-energy tensor for different time steps
    FILE* finout;
    finout = fopen("phiphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.100f ", bi_linears->phi_phi[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("DphiDphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout1, "%.100f ", bi_linears->Dphi_Dphi[m] );
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("DphiKphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout2, "%.100f ", bi_linears->Dphi_Kphi[m] );
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("KphiKphi.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout3, "%.100f ", bi_linears->Kphi_Kphi[m] );
    }
    fclose(finout3);

    FILE* finout4;
    finout4 = fopen("phithetaphitheta.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout4, "%.100f ", bi_linears->del_theta_phi_del_theta_phi[m]);
    }
    fclose(finout4);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that sets the functions to zero */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_zero(Fields* fields) {

    for (int i = 0; i < v_size; ++i) {
        fields->phi[i] = 0;
        fields->r[i] = 0;
        fields->sigma[i] = 0;
        fields->D_sigma[i] = 0;
        fields->D_phi[i] = 0;
        fields->K_phi[i] = 0;
        fields->K_r[i] = 0;
        fields->D_r[i] = 0;
        fields->mu[i] = 0;
        fields->lambda[i] = 0;
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that frees up memory */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void free_memory(Fields* fields, Q_Fields** q_fields) {
    free(fields->phi);
    free(fields->r);
    free(fields->sigma);
    free(fields);
}
/* Main */
void main() {

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* DEFINE VARIABLES AND ASSIGN ALL THE MEMORY *//////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Fields* fields;
    Fields* fields_n;
    Fields* fields_nm1;
    Fields* fields_nm2;

    Q_Fields** q_fields;
    Q_Fields** q_fields_n;
    Q_Fields** q_fields_nm1;
    Q_Fields** q_fields_nm2;

    Bi_Linears* bi_linears;
    Bi_Linears* bi_linears_n;
    Bi_Linears* bi_linears_nm1;
    Bi_Linears* bi_linears_nm2;

    fields = (Fields*)malloc(sizeof(Fields));
    fields->phi = (double*)malloc(v_size * sizeof(double));
    fields->D_phi = (double*)malloc(v_size * sizeof(double));
    fields->K_phi = (double*)malloc(v_size * sizeof(double));
    fields->r = (double*)malloc(v_size * sizeof(double));
    fields->K_r = (double*)malloc(v_size * sizeof(double));
    fields->D_r = (double*)malloc(v_size * sizeof(double));
    fields->sigma = (double*)malloc(v_size * sizeof(double));
    fields->D_sigma = (double*)malloc(v_size * sizeof(double));
    fields->lambda = (double*)malloc(v_size * sizeof(double));
    fields->mu = (double*)malloc(v_size * sizeof(double));


    fields_n = (Fields*)malloc(sizeof(Fields));
    fields_n->phi = (double*)malloc(v_size * sizeof(double));
    fields_n->D_phi = (double*)malloc(v_size * sizeof(double));
    fields_n->K_phi = (double*)malloc(v_size * sizeof(double));
    fields_n->r = (double*)malloc(v_size * sizeof(double));
    fields_n->K_r = (double*)malloc(v_size * sizeof(double));
    fields_n->D_r = (double*)malloc(v_size * sizeof(double));
    fields_n->sigma = (double*)malloc(v_size * sizeof(double));
    fields_n->D_sigma = (double*)malloc(v_size * sizeof(double));
    fields_n->lambda = (double*)malloc(v_size * sizeof(double));
    fields_n->mu = (double*)malloc(v_size * sizeof(double));

    fields_nm1 = (Fields*)malloc(sizeof(Fields));
    fields_nm1->phi = (double*)malloc(v_size * sizeof(double));
    fields_nm1->D_phi = (double*)malloc(v_size * sizeof(double));
    fields_nm1->K_phi = (double*)malloc(v_size * sizeof(double));
    fields_nm1->r = (double*)malloc(v_size * sizeof(double));
    fields_nm1->K_r = (double*)malloc(v_size * sizeof(double));
    fields_nm1->D_r = (double*)malloc(v_size * sizeof(double));
    fields_nm1->sigma = (double*)malloc(v_size * sizeof(double));
    fields_nm1->D_sigma = (double*)malloc(v_size * sizeof(double));
    fields_nm1->lambda = (double*)malloc(v_size * sizeof(double));
    fields_nm1->mu = (double*)malloc(v_size * sizeof(double));

    fields_nm2 = (Fields*)malloc(sizeof(Fields));
    fields_nm2->phi = (double*)malloc(v_size * sizeof(double));
    fields_nm2->D_phi = (double*)malloc(v_size * sizeof(double));
    fields_nm2->K_phi = (double*)malloc(v_size * sizeof(double));
    fields_nm2->r = (double*)malloc(v_size * sizeof(double));
    fields_nm2->K_r = (double*)malloc(v_size * sizeof(double));
    fields_nm2->D_r = (double*)malloc(v_size * sizeof(double));
    fields_nm2->sigma = (double*)malloc(v_size * sizeof(double));
    fields_nm2->D_sigma = (double*)malloc(v_size * sizeof(double));
    fields_nm2->lambda = (double*)malloc(v_size * sizeof(double));
    fields_nm2->mu = (double*)malloc(v_size * sizeof(double));

    bi_linears = (Bi_Linears*)malloc(sizeof(Bi_Linears));
    bi_linears->phi_phi = (double*)malloc(v_size * sizeof(double));
    bi_linears->Dphi_Dphi = (double*)malloc(v_size * sizeof(double));
    bi_linears->Dphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears->Kphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears->del_theta_phi_del_theta_phi = (double*)malloc(v_size * sizeof(double));

    bi_linears_n = (Bi_Linears*)malloc(sizeof(Bi_Linears));
    bi_linears_n->phi_phi = (double*)malloc(v_size * sizeof(double));
    bi_linears_n->Dphi_Dphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_n->Dphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_n->Kphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_n->del_theta_phi_del_theta_phi = (double*)malloc(v_size * sizeof(double));

    bi_linears_nm1 = (Bi_Linears*)malloc(sizeof(Bi_Linears));
    bi_linears_nm1->phi_phi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm1->Dphi_Dphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm1->Dphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm1->Kphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm1->del_theta_phi_del_theta_phi = (double*)malloc(v_size * sizeof(double));

    bi_linears_nm2 = (Bi_Linears*)malloc(sizeof(Bi_Linears));
    bi_linears_nm2->phi_phi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm2->Dphi_Dphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm2->Dphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm2->Kphi_Kphi = (double*)malloc(v_size * sizeof(double));
    bi_linears_nm2->del_theta_phi_del_theta_phi = (double*)malloc(v_size * sizeof(double));


    q_fields = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));
    q_fields_n = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));
    q_fields_nm1 = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));
    q_fields_nm2 = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

        q_fields[which_q_field] = (Q_Fields*)malloc(sizeof(Q_Fields));
        q_fields_n[which_q_field] = (Q_Fields*)malloc(sizeof(Q_Fields));
        q_fields_nm1[which_q_field] = (Q_Fields*)malloc(sizeof(Q_Fields));
        q_fields_nm2[which_q_field] = (Q_Fields*)malloc(sizeof(Q_Fields));


        q_fields[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields[which_q_field]->D_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields[which_q_field]->K_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields[which_q_field]->mu = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_n[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_n[which_q_field]->D_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_n[which_q_field]->K_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_n[which_q_field]->mu = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_nm1[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm1[which_q_field]->D_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm1[which_q_field]->K_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm1[which_q_field]->mu = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_nm2[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm2[which_q_field]->D_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm2[which_q_field]->K_phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_nm2[which_q_field]->mu = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        for (int k = 0; k < number_of_k_modes; k++) {

            q_fields[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields[which_q_field]->D_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields[which_q_field]->K_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields[which_q_field]->mu[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_n[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_n[which_q_field]->D_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_n[which_q_field]->K_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_n[which_q_field]->mu[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_nm1[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm1[which_q_field]->D_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm1[which_q_field]->K_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm1[which_q_field]->mu[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_nm2[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm2[which_q_field]->D_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm2[which_q_field]->K_phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_nm2[which_q_field]->mu[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            for (int l = 0; l < number_of_l_modes; ++l) {

                q_fields[which_q_field]->phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields[which_q_field]->D_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields[which_q_field]->K_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields[which_q_field]->mu[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));

                q_fields_n[which_q_field]->phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_n[which_q_field]->D_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_n[which_q_field]->K_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_n[which_q_field]->mu[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));

                q_fields_nm1[which_q_field]->phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm1[which_q_field]->D_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm1[which_q_field]->K_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm1[which_q_field]->mu[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));

                q_fields_nm2[which_q_field]->phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm2[which_q_field]->D_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm2[which_q_field]->K_phi[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_nm2[which_q_field]->mu[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
            }
        }
    }
    
    double** full_r;
    full_r = (double**)malloc(N_HALF * (u_size - 1) / 2 * sizeof(double*));
    for (int i = 0; i < N_HALF * (u_size - 1) / 2; i++) {
        full_r[i] = (double*)malloc(v_size * sizeof(double));
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* ACTUAL EVOLUTION OF FIELDS *//////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double v[v_size];
    make_points(0, v);
    save_points(v);
    double cosm_const=0.0;
    set_zero(fields);
    set_zero(fields_n);
    set_zero(fields_nm1);
    set_zero(fields_nm2);


    charac_initial_conditions(fields);

    charac_initial_conditions_quantum(q_fields);

    set_minkowski(fields);

    cosm_const = set_cosm_constant(fields, q_fields, bi_linears);
    printf("cosmological constant = %.5f, \n", cosm_const);

    v_evolution_initial(cosm_const, fields, q_fields, bi_linears);

    full_evolution(cosm_const, full_r, fields, fields_n, fields_nm1, fields_nm2, q_fields, q_fields_n, q_fields_nm1, q_fields_nm2, bi_linears, bi_linears_n, bi_linears_nm1, bi_linears_nm2);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* SAVE FIELDS */////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    save_Dphiq(q_fields[0]->D_phi[0][0]);
    save_Kphiq(q_fields[0]->K_phi[0][0]);
    save_muphiq(q_fields[0]->mu[0][0]);

    save_Dphi(fields->D_phi);
    save_Kphi(fields->K_phi);
    save_muphi(fields->mu);

    save_r1(fields->r);
    save_phi(fields->phi);
    save_sigma(fields->sigma);
    save_r(full_r);
    save_vmat();
    save_u();
    printf("done");


    save_bi_linears(0, cosm_const, fields, q_fields, bi_linears);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* FREE UP ALL THE MEMORY *//////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    free(fields->phi);
    free(fields->D_phi);
    free(fields->K_phi);
    free(fields->r);
    free(fields->D_r);
    free(fields->K_r);
    free(fields->sigma);
    free(fields->D_sigma);
    free(fields->mu);
    free(fields->lambda);
    free(fields);

    free(fields_n->phi);
    free(fields_n->D_phi);
    free(fields_n->K_phi);
    free(fields_n->r);
    free(fields_n->D_r);
    free(fields_n->K_r);
    free(fields_n->sigma);
    free(fields_n->D_sigma);
    free(fields_n->mu);
    free(fields_n->lambda);
    free(fields_n);

    free(fields_nm1->phi);
    free(fields_nm1->D_phi);
    free(fields_nm1->K_phi);
    free(fields_nm1->r);
    free(fields_nm1->D_r);
    free(fields_nm1->K_r);
    free(fields_nm1->sigma);
    free(fields_nm1->D_sigma);
    free(fields_nm1->mu);
    free(fields_nm1->lambda);
    free(fields_nm1);

    free(fields_nm2->phi);
    free(fields_nm2->D_phi);
    free(fields_nm2->K_phi);
    free(fields_nm2->r);
    free(fields_nm2->D_r);
    free(fields_nm2->K_r);
    free(fields_nm2->sigma);
    free(fields_nm2->D_sigma);
    free(fields_nm2->mu);
    free(fields_nm2->lambda);
    free(fields_nm2);

    free(bi_linears->phi_phi);
    free(bi_linears->Dphi_Dphi);
    free(bi_linears->Kphi_Kphi);
    free(bi_linears->Dphi_Kphi);
    free(bi_linears->del_theta_phi_del_theta_phi);
    free(bi_linears);

    free(bi_linears_n->phi_phi);
    free(bi_linears_n->Dphi_Dphi);
    free(bi_linears_n->Kphi_Kphi);
    free(bi_linears_n->Dphi_Kphi);
    free(bi_linears_n->del_theta_phi_del_theta_phi);
    free(bi_linears_n);

    free(bi_linears_nm1->phi_phi);
    free(bi_linears_nm1->Dphi_Dphi);
    free(bi_linears_nm1->Kphi_Kphi);
    free(bi_linears_nm1->Dphi_Kphi);
    free(bi_linears_nm1->del_theta_phi_del_theta_phi);
    free(bi_linears_nm1);

    free(bi_linears_nm2->phi_phi);
    free(bi_linears_nm2->Dphi_Dphi);
    free(bi_linears_nm2->Kphi_Kphi);
    free(bi_linears_nm2->Dphi_Kphi);
    free(bi_linears_nm2->del_theta_phi_del_theta_phi);
    free(bi_linears_nm2);

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(q_fields[which_q_field]->phi[k][l]);
                free(q_fields[which_q_field]->D_phi[k][l]);
                free(q_fields[which_q_field]->K_phi[k][l]);
                free(q_fields[which_q_field]->mu[k][l]);

                free(q_fields_n[which_q_field]->phi[k][l]);
                free(q_fields_n[which_q_field]->D_phi[k][l]);
                free(q_fields_n[which_q_field]->K_phi[k][l]);
                free(q_fields_n[which_q_field]->mu[k][l]);

                free(q_fields_nm1[which_q_field]->phi[k][l]);
                free(q_fields_nm1[which_q_field]->D_phi[k][l]);
                free(q_fields_nm1[which_q_field]->K_phi[k][l]);
                free(q_fields_nm1[which_q_field]->mu[k][l]);

                free(q_fields_nm2[which_q_field]->phi[k][l]);
                free(q_fields_nm2[which_q_field]->D_phi[k][l]);
                free(q_fields_nm2[which_q_field]->K_phi[k][l]);
                free(q_fields_nm2[which_q_field]->mu[k][l]);
            }
            free(q_fields[which_q_field]->phi[k]);
            free(q_fields[which_q_field]->D_phi[k]);
            free(q_fields[which_q_field]->K_phi[k]);
            free(q_fields[which_q_field]->mu[k]);

            free(q_fields_n[which_q_field]->phi[k]);
            free(q_fields_n[which_q_field]->D_phi[k]);
            free(q_fields_n[which_q_field]->K_phi[k]);
            free(q_fields_n[which_q_field]->mu[k]);

            free(q_fields_nm1[which_q_field]->phi[k]);
            free(q_fields_nm1[which_q_field]->D_phi[k]);
            free(q_fields_nm1[which_q_field]->K_phi[k]);
            free(q_fields_nm1[which_q_field]->mu[k]);

            free(q_fields_nm2[which_q_field]->phi[k]);
            free(q_fields_nm2[which_q_field]->D_phi[k]);
            free(q_fields_nm2[which_q_field]->K_phi[k]);
            free(q_fields_nm2[which_q_field]->mu[k]);

        }
        free(q_fields[which_q_field]->phi);
        free(q_fields[which_q_field]->D_phi);
        free(q_fields[which_q_field]->K_phi);
        free(q_fields[which_q_field]->mu);

        free(q_fields_n[which_q_field]->phi);
        free(q_fields_n[which_q_field]->D_phi);
        free(q_fields_n[which_q_field]->K_phi);
        free(q_fields_n[which_q_field]->mu);

        free(q_fields_nm1[which_q_field]->phi);
        free(q_fields_nm1[which_q_field]->D_phi);
        free(q_fields_nm1[which_q_field]->K_phi);
        free(q_fields_nm1[which_q_field]->mu);

        free(q_fields_nm2[which_q_field]->phi);
        free(q_fields_nm2[which_q_field]->D_phi);
        free(q_fields_nm2[which_q_field]->K_phi);
        free(q_fields_nm2[which_q_field]->mu);

        free(q_fields[which_q_field]);
        free(q_fields_n[which_q_field]);
        free(q_fields_nm1[which_q_field]);
        free(q_fields_nm2[which_q_field]);
    }
    free(q_fields);
    free(q_fields_n);
    free(q_fields_nm1);
    free(q_fields_nm2);


    for (int i = 0; i < u_size; i++) {
        free(full_r[i]);
    }
    free(full_r);

    printf("done");


}
