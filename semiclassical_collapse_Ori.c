
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
#define         u_size                             1001                                                                                   // size of grid in u direction
#define         v_size                             4001                                                                                   // size of grid in v direction
#define         du                                 0.02                                                                                  // step size in u direction
#define         dv                                 0.02                                                                                  // step size in v direction
#define         u_max                              du*(u_size-1)
#define         v_max                              dv*(v_size-1)
#define         r0                                 10.0

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
double          massSq[number_of_q_fields] = { muSq, mSqGhost, 3.0 * mSqGhost, mSqGhost, 3.0 * mSqGhost, 4.0 * mSqGhost };       // masses of the ghost fields
double          ghost_or_physical[6] = { 1 , -1 , 1 , -1 , 1, -1 };                                                        // distinguishing between the real and ghost fields

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
};
typedef struct fields Fields;

struct q_fields {
    __complex__ double*** phi;
};
typedef struct q_fields Q_Fields;

struct initial_fields {
    double* phi_u0;
    double* sigma_u0;
    double* r_u0;
    double* phi_v0;
    double* sigma_v0;
    double* r_v0;
};
typedef struct initial_fields Initial_Fields;

struct initial_q_fields {
    __complex__ double*** phi_u0;
    __complex__ double*** phi_v0;
};
typedef struct initial_q_fields Initial_Q_Fields;

struct field_dot {
    double phi;
    double sigma;
    double r;
};
typedef struct field_dot Field_Dot;

struct bi_linears {
    double* phi_phi;
};
typedef struct bi_linears Bi_Linears;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Making the points for the spatial grid*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void make_points(double v[v_size]) {
    
    for (int i = 0; i < v_size; ++i) {
        v[i] = i * dv;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Save points and fields */
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
    for (int m = 0; m < u_size; ++m) {
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
void save_phi(double* field) {
    FILE* finout;
    finout = fopen("phiphi_l100k50_mghost50.txt", "w");
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
void save_constrv(double constr[v_size]) {
    FILE* finout;
    finout = fopen("constr_nullv0.txt", "w");
    for (int m = 0; m < v_size; ++m) {
        fprintf(finout, "%.20f ", constr[m]);
    }
    fclose(finout);
}
void save_constru(double constr[u_size]) {
    FILE* finout;
    finout = fopen("constr_nullu0.txt", "w");
    for (int m = 0; m < u_size; ++m) {
        fprintf(finout, "%.20f ", constr[m]);
    }
    fclose(finout);
}
void save_r(double** field) {

    FILE* finout;
    finout = fopen("r_long_024.txt", "w");
    for (int n = 0; n < u_size; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < v_size; ++m) {
            fprintf(finout, "%.200f ", field[n][m]);
        }
    }
    fclose(finout);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Define differentiation functions */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv_vo2(double x1,double x2,double x3,double x4) {
    double der = 0.0;
    return der = (x2 - x1 + x4  - x3) / (2.0 * dv);
}
double first_deriv_uo2(double x1, double x2, double x3, double x4) {
    double der = 0.0;
    return der = (x3 - x1 + x4 - x2) / (2.0 * du);
}

double first_deriv_vo1(double x1, double x2) {
    double der = 0.0;
    return der = (x2 - x1) / (1.0*dv);
}
double first_deriv_uo1(double x1, double x3) {
    double der = 0.0;
    return der = (x3 - x1) / (1.0*du);
}
double first_derivv(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m + 1] - field[m]) / (dv);
    }
    if (m == v_size - 1) {
        return der = (field[m] - field[m-1]) / (dv);
    }
    if (m == 1 || m == v_size - 2) {
        return der = (field[m + 1] - field[m - 1]) / (2.0*dv);
    }
    if (m > 1 && m < v_size - 2) {
        return der = (-field[m+2] + 8.0*field[m + 1] - 8.0*field[m - 1] + field[m - 2]) / (12.0 * dv);
    }
}
double first_derivu(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m + 1] - field[m]) / (du);
    }
    if (m == u_size - 1) {
        return der = (field[m] - field[m - 1]) / (du);
    }
    if (m == 1 || m == u_size - 2) {
        return der = (field[m + 1] - field[m - 1]) / (2.0 * du);
    }
    if (m > 1 && m < u_size - 2) {
        return der = (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 2]) / (12.0 * du);
    }
}
double second_derivv(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m+1] + field[m +2]) / (dv * dv);
    }
    if (m == v_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (dv * dv);
    }
    if (m == 1 || m == v_size - 2) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (dv * dv);
    }
    if (m > 1 && m < v_size - 2) {
        return der = (-field[m + 2] + 16.0 * field[m + 1] - 30.0 * field[m] + 16.0 * field[m - 1] - field[m - 2]) / (12.0 * dv*dv);
    }
}
double second_derivu(int m, double* field) {
    double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m+1] + field[m +2]) / (du * du);
    }
    if (m == u_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (du * du);
    }
    if (m == 1 || m == u_size - 2) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (du * du);
    }
    if (m > 1 && m < u_size - 2) {
        return der = (-field[m + 2] + 16.0 * field[m + 1] - 30.0 * field[m] + 16.0 * field[m - 1] - field[m - 2]) / (12.0 * du*du);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Define differentiation functions for complex fields */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double first_deriv_vo2_comp(__complex__ double x1, __complex__ double x2, __complex__ double x3, __complex__ double x4) {
    __complex__ double der = 0.0;
    return der = (x2 - x1 + x4 - x3) / (2.0 * dv);
}
__complex__ double first_deriv_uo2_comp(__complex__ double x1, __complex__ double x2, __complex__ double x3, __complex__ double x4) {
    __complex__ double der = 0.0;
    return der = (x3 - x1 + x4 - x2) / (2.0 * du);
}

__complex__ double first_deriv_vo1_comp(__complex__ double x1, __complex__ double x2) {
    __complex__ double der = 0.0;
    return der = (x2 - x1) / (1.0 * dv);
}
__complex__ double first_deriv_uo1_comp(__complex__ double x1, __complex__ double x3) {
    __complex__ double der = 0.0;
    return der = (x3 - x1) / (1.0 * du);
}
__complex__ double first_derivv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        return der = (field[m + 1] - field[m]) / (dv);
    }
    if (m == v_size - 1) {
        return der = (field[m] - field[m - 1]) / (dv);
    }
    if (m == 1 || m == v_size - 2) {
        return der = (field[m + 1] - field[m - 1]) / (2.0 * dv);
    }
    if (m > 1 && m < v_size - 2) {
        return der = (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 2]) / (12.0 * dv);
    }
}
__complex__ double first_derivu_comp(int m, __complex__  double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        return der = (field[m + 1] - field[m]) / (du);
    }
    if (m == u_size - 1) {
        return der = (field[m] - field[m - 1]) / (du);
    }
    if (m == 1 || m == u_size - 2) {
        return der = (field[m + 1] - field[m - 1]) / (2.0 * du);
    }
    if (m > 1 && m < u_size - 2) {
        return der = (-field[m + 2] + 8.0 * field[m + 1] - 8.0 * field[m - 1] + field[m - 2]) / (12.0 * du);
    }
}
__complex__ double second_derivv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m + 1] + field[m + 2]) / (dv * dv);
    }
    if (m == v_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (dv * dv);
    }
    if (m == 1 || m == v_size - 2) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (dv * dv);
    }
    if (m > 1 && m < v_size - 2) {
        return der = (-field[m + 2] + 16.0 * field[m + 1] - 30.0 * field[m] + 16.0 * field[m - 1] - field[m - 2]) / (12.0 * dv * dv);
    }
}
__complex__ double second_derivu_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;
    if (m == 0) {
        return der = (field[m] - 2.0 * field[m + 1] + field[m + 2]) / (du * du);
    }
    if (m == u_size - 1) {
        return der = (field[m - 2] - 2.0 * field[m - 1] + field[m]) / (du * du);
    }
    if (m == 1 || m == u_size - 2) {
        return der = (field[m + 1] - 2.0 * field[m] + field[m - 1]) / (du * du);
    }
    if (m > 1 && m < u_size - 2) {
        return der = (-field[m + 2] + 16.0 * field[m + 1] - 30.0 * field[m] + 16.0 * field[m - 1] - field[m - 2]) / (12.0 * du * du);
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
__complex__ double phi_mode_profile(double k, int l, double r, double t) {
    
    return sqrt(k / PI) * cexp(-I * k * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l);
}

__complex__ double phi_mode_profile_massive(double msq, double k, int l, double r, double t) {
    
    return k / sqrt(PI * sqrt(k * k + msq)) * cexp(-I * sqrt(k * k + msq) * t) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function to find the initial r on both characteristic surfaces */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_r_u0(Initial_Fields* init_fields) {
    double r_im1, r_i, r_primeprime, phi_prime, v;
    init_fields->r_u0[0] = r0;
    init_fields->r_u0[1] = r0 + 0.5 * dv;
    r_im1 = r0;
    r_i   = r0 + 0.5 * dv;

    for (int i = 1; i < v_size-1; i++) {
        v = i * dv;
        phi_prime = -( (v-initial_radius) / (initial_width*initial_width)) * init_fields->phi_u0[i];
        r_primeprime = r_i * phi_prime * phi_prime;

        init_fields->r_u0[i + 1] = 2.0 * r_i - r_im1 - dv * dv * r_primeprime;

        r_im1 = r_i;
        r_i = init_fields->r_u0[i + 1];
    }
}
void find_r_v0(int n, Initial_Fields* init_fields) {
    double r_im1, r_i, sigma_prime;
    
    r_im1 = init_fields->r_v0[n-2];
    r_i = init_fields->r_v0[n -1];
    sigma_prime = (init_fields->sigma_v0[n-1] - init_fields->sigma_v0[n-2])/du;

    init_fields->r_v0[n] = 1.0/(1.0-0.5*du*sigma_prime)*(2.0*r_i - r_im1* (1.0 + 0.5 * du * sigma_prime));

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initial conditions for all fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void charac_initial_conditions(Initial_Fields* init_fields, Initial_Fields* init_fields_test) {
    double v[v_size];
    make_points(v);

    /* initial v slice */
    for (int i = 0; i < v_size; ++i) {
        init_fields->sigma_u0[i] = 0.0;
        init_fields->phi_u0[i] = amplitude * exp(-1.0 / 2.0 * pow((v[i] - initial_radius) / initial_width, 2.0));

        init_fields_test->phi_u0[i] = exp(-1.0 / 2.0 * pow((v[i] - initial_radius_test) / initial_width_test, 2.0));

    }

    /* initial u slice */
    for (int i = 0; i < u_size; ++i) {
        init_fields->sigma_v0[i] = 0.0;  // IMPORTANT GAUGE CHOICE, atm this is the STANDARD choice
        init_fields->phi_v0[i] = 0.0;
        init_fields->r_v0[i] = r0 - 0.5 * i * du;

        init_fields_test->phi_v0[i] = exp(-1.0 / 2.0 * pow((du*i - initial_radius_test_u) / initial_width_test_u, 2.0));

    }

    /* now let's determine the initial r (from constraint eqs)*/
    find_r_u0(init_fields);
    
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initial conditions for the quantum mode functions */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void charac_initial_conditions_quantum(Initial_Q_Fields** init_q_fields, Initial_Fields* init_fields) {
    double v[v_size];
    make_points(v);

   
    double k_wavenumber, omega_phi;
    int l_value;
    for (int k = 0; k < number_of_k_modes; ++k) {
        k_wavenumber = (k_start + (k + 1)) * k_min;
        for (int l = 0; l < number_of_l_modes; ++l) {
            l_value = l_start + l * l_step;
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                omega_phi = sqrt(k_wavenumber * k_wavenumber + massSq[which_q_field]);
                if (massSq[which_q_field] == 0) {
                    /* initial v slice */
                    for (int i = 0; i < v_size; ++i) {
                        double vv, uu;
                        vv = i * dv;
                        uu = 0.0;
                        double r = init_fields->r_u0[i];//  r0 + (vv - uu) / 2.0;
                        double t = init_fields->r_u0[i]-r0; //(vv + uu) / 2.0;

                        init_q_fields[which_q_field]->phi_u0[k][l][i] = phi_mode_profile(k_wavenumber, l_value, r, t);
                    }
                    /* initial u slice */
                    for (int i = 0; i < u_size; ++i) {
                        double vv, uu;
                        vv = 0.0;
                        uu = i * du;
                        double r = r0 + (vv - uu) / 2.0;
                        double t = (vv + uu) / 2.0;

                        init_q_fields[which_q_field]->phi_v0[k][l][i] = phi_mode_profile(k_wavenumber, l_value, r, t);
                    }
                }
                else {
                    /* initial v slice */
                    for (int i = 0; i < v_size; ++i) {
                        double vv, uu;
                        vv = i * dv;
                        uu = 0.0;
                        double r = init_fields->r_u0[i];//  r0 + (vv - uu) / 2.0;
                        double t = init_fields->r_u0[i] - r0; //(vv + uu) / 2.0;

                        init_q_fields[which_q_field]->phi_u0[k][l][i] = phi_mode_profile_massive(massSq[which_q_field], k_wavenumber, l_value, r, t);
                    }
                    /* initial u slice */
                    for (int i = 0; i < u_size; ++i) {
                        double vv, uu;
                        vv = 0.0;
                        uu = i * du;
                        double r = r0 + (vv - uu) / 2.0;
                        double t = (vv + uu) / 2.0;

                        init_q_fields[which_q_field]->phi_v0[k][l][i] = phi_mode_profile_massive(massSq[which_q_field], k_wavenumber, l_value, r, t);
                    }
                }


            }
        }
    }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Set up the variables on the u=0 null slice */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_u0_slice(Initial_Fields* init_fields, Fields* fields, Initial_Fields* init_fields_test, Fields* fields_test, Initial_Q_Fields** init_q_fields, Q_Fields** q_fields) {
    for (int i = 0; i < v_size; ++i) {
        fields->phi[i]   = init_fields->phi_u0[i];
        fields->r[i]     = init_fields->r_u0[i];
        fields->sigma[i] = init_fields->sigma_u0[i];

        fields_test->phi[i] = init_fields_test->phi_u0[i];

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                    q_fields[which_q_field]->phi[k][l][i] = init_q_fields[which_q_field]->phi_u0[k][l][i];

                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* These functions check the contraint equations */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void check_constrsv(double constr[v_size],Fields* fields) {
    
    for (int i = 0; i < v_size; i++) {
        constr[i] = cabs(second_derivv(i, fields->r) - first_derivv(i, fields->r) * first_derivv(i, fields->sigma) + fields->r[i] * pow( first_derivv(i, fields->phi) , 2.0));
        //constr[i] = constr[i] / (fields->r[i] * pow(first_derivv(i, fields->phi), 2.0));
    }
}
void check_constrsu(double constr[u_size],Fields* fields) {
    
    for (int i = 0; i < u_size; i++) {
        constr[i] = cabs(second_derivu(i, fields->r) - first_derivu(i, fields->r) * first_derivu(i, fields->sigma) + fields->r[i] * pow( first_derivu(i, fields->phi) , 2.0));
        //constr[i] = constr[i] / (fields->r[i] * pow(first_derivu(i, fields->phi), 2.0));

    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Set old version of functions before they are evolved to the next time step */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_fields_old(Fields* fields, Fields* fields_old, Fields* fields_test, Fields* fields_test_old, Q_Fields** q_fields, Q_Fields** q_fields_old) {

    for (int i = 0; i < v_size; ++i) {
        fields_old->phi[i]   = fields->phi[i];
        fields_old->r[i]     = fields->r[i];
        fields_old->sigma[i] = fields->sigma[i];

        fields_test_old->phi[i] = fields_test->phi[i];

        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                    q_fields_old[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i];

                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Set boundary conditions for the evolution (based on the v=0 null slice) */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_boundary_conds(int n, Initial_Fields* init_fields, Fields* fields, Initial_Fields* init_fields_test, Fields* fields_test, Initial_Q_Fields** init_q_fields, Q_Fields** q_fields) {
    
    fields->phi[0]   = init_fields->phi_v0[n];
    fields->r[0]     = init_fields->r_v0[n];
    fields->sigma[0] = init_fields->sigma_v0[n];

    fields_test->phi[0] = init_fields_test->phi_v0[n];

    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                q_fields[which_q_field]->phi[k][l][0] = init_q_fields[which_q_field]->phi_v0[k][l][n];

            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Calculate the norm */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double norm(__complex__ double number) {
    double nor = 0.0;
    nor = (pow((__real__ number), 2.0) + pow((__imag__ number), 2.0));
    return nor;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Finds the fluctuations around the connected two point function */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double find_phi_phi(int n, int i, Fields* fields, Q_Fields** q_fields) {
    double r, r_l, t;
    double phi_phi;
    __complex__ double Phi_mode, Chi_mode;
    int l_value;

    phi_phi = 0.0;

    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                l_value = l_start + l * l_step;

                r = fields->r[i];
                r_l = pow(r, l_value);
                //t = r - r0;
                /* PHI MODE */
                Phi_mode = r_l * (q_fields[which_q_field]->phi[k][l][i]);
                if (l_value == 0) {
                    Chi_mode = first_derivv_comp(i, q_fields[which_q_field]->phi[k][l]);
                }
                else if (l_value == 1) {
                    Chi_mode = first_derivv(i, fields->r) * q_fields[which_q_field]->phi[k][l][i] + r * first_derivv_comp(i, q_fields[which_q_field]->phi[k][l]);
                }
                else {
                    Chi_mode = l_value * pow(r, l_value - 1) * first_derivv(i, fields->r) * q_fields[which_q_field]->phi[k][l][i] + r_l * first_derivv_comp(i, q_fields[which_q_field]->phi[k][l]);
                }
                /* ACTUAL BILINEARS */
                phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode);

            }
        }
    }
    
    return phi_phi;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Diamond evolution of all fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void diamond_evolution(int n, Initial_Fields* init_fields, Fields* fields, Fields* fields_old, Initial_Fields* init_fields_test, Fields* fields_test, Fields* fields_test_old, Initial_Q_Fields** init_q_fields, Q_Fields** q_fields, Q_Fields** q_fields_old) {




    /* start with r */
    fields->r[0] = r0 - 0.5 * (n + 1) * du; 

    for (int i = 1; i < v_size; ++i) {
        double r_dot;
        double r, sigma, r_uderiv, r_vderiv;
        double r1, r2, r3, r4tilda;
        double sigma2, sigma3;

        // predictor stage
        r1 = fields_old->r[i-1];
        r2 = fields_old->r[i];
        r3 = fields->r[i - 1];
        sigma2 = fields_old->sigma[i];
        sigma3 = fields->sigma[i-1];


        r        = (r2 + r3) / 2.0;
        sigma    = (sigma2 + sigma3) / 2.0;
        r_uderiv = first_deriv_uo1(r1, r3);
        r_vderiv = first_deriv_vo1(r1, r2);

        r_dot = (r!=0 ? -r_uderiv * r_vderiv / r - exp(sigma) / (4.0 * r) : 0.0);

        r4tilda = r3 + r2 - r1 + (du * dv) * r_dot;

        // corrector stage  
        r_uderiv = first_deriv_uo2(r1, r2, r3, r4tilda);
        r_vderiv = first_deriv_vo2(r1, r2, r3, r4tilda);

        r_dot =(r!=0 ? -r_uderiv * r_vderiv / r - exp(sigma) / (4.0 * r) : 0.0);

        fields->r[i] = r3 + r2 - r1 + (du * dv) * r_dot;
        
    }
    

    /* then phi */
    
    for (int i = 1; i < v_size; ++i) {
        double phi_dot;
        double phi, phi_uderiv, phi_vderiv;
        double r, r_uderiv, r_vderiv;
        double r1, r2, r3, r4;
        double phi1, phi2, phi3, phi4tilda;

        // predictor stage
        r1 = fields_old->r[i - 1];
        r2 = fields_old->r[i];
        r3 = fields->r[i - 1];
        r4 = fields->r[i];
        phi1 = fields_old->phi[i - 1];
        phi2 = fields_old->phi[i];
        phi3 = fields->phi[i - 1];

        r = (r2 + r3) / 2.0;
        r_uderiv = first_deriv_uo2(r1, r2, r3, r4);
        r_vderiv = first_deriv_vo2(r1, r2, r3, r4);

        phi = (phi2 + phi3) / 2.0;
        phi_uderiv = first_deriv_uo1(phi1, phi3);
        phi_vderiv = first_deriv_vo1(phi1, phi2);

        phi_dot = (r != 0 ? -1.0 / r * (r_uderiv * phi_vderiv + r_vderiv * phi_uderiv) : 0.0);

        phi4tilda = phi3 + phi2 - phi1 + du * dv * phi_dot;

        // corrector stage  
        phi_uderiv = first_deriv_uo2(phi1, phi2, phi3, phi4tilda);
        phi_vderiv = first_deriv_vo2(phi1, phi2, phi3, phi4tilda);

        phi_dot = (r != 0 ? -1.0 / r * (r_uderiv * phi_vderiv + r_vderiv * phi_uderiv) : 0.0);

        fields->phi[i] = phi3 + phi2 - phi1 + du * dv * phi_dot;
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    double k_wavenumber = (k_start + (k + 1)) * k_min;
                    double l_val = l_start + l * l_step;
                    double sigma;
                    __complex__ double qphi_dot;
                    __complex__ double qphi, qphi_uderiv, qphi_vderiv;
                    __complex__ double qphi1, qphi2, qphi3, qphi4tilda;

                    sigma  = (fields_old->sigma[i] + fields->sigma[i - 1]) / 2.0;
                    // predictor stage
                    qphi1 = q_fields_old[which_q_field]->phi[k][l][i - 1]; 
                    qphi2 = q_fields_old[which_q_field]->phi[k][l][i];
                    qphi3 = q_fields[which_q_field]->phi[k][l][i - 1];

                    qphi = (qphi2 + qphi3) / 2.0;
                    qphi_uderiv = first_deriv_uo1_comp(qphi1, qphi3);
                    qphi_vderiv = first_deriv_vo1_comp(qphi1, qphi2);

                    qphi_dot = -(l_val + 1.0) / (r) * (r_uderiv * qphi_vderiv + r_vderiv * qphi_uderiv)
                        - (l_val * l_val) / (r * r) * (exp(sigma) / 4.0 + r_uderiv * r_vderiv) * qphi
                        - 0.25* massSq[which_q_field] * exp(sigma) * qphi;

                    qphi4tilda = qphi3 + qphi2 - qphi1 + du * dv * qphi_dot;

                    // corrector stage  
                    qphi_uderiv = first_deriv_uo2_comp(qphi1, qphi2, qphi3, qphi4tilda);
                    qphi_vderiv = first_deriv_vo2_comp(qphi1, qphi2, qphi3, qphi4tilda);

                    qphi_dot = -(l_val + 1.0) / (r) * (r_uderiv * qphi_vderiv + r_vderiv * qphi_uderiv)
                        - (l_val * l_val) / (r * r) * (exp(sigma) / 4.0 + r_uderiv * r_vderiv) * qphi
                        - 0.25*massSq[which_q_field] * exp(sigma) * qphi;

                    q_fields[which_q_field]->phi[k][l][i] = qphi3 + qphi2 - qphi1 + du * dv * qphi_dot;

                }
            }
        }
    }
    for (int i = 1; i < v_size; ++i) {
        double phi_dot;
        double phi, phi_uderiv, phi_vderiv;
        double r, r_uderiv, r_vderiv;
        double r1, r2, r3, r4;
        double phi1, phi2, phi3, phi4tilda;

        // predictor stage
        r1 = fields_old->r[i - 1];
        r2 = fields_old->r[i];
        r3 = fields->r[i - 1];
        r4 = fields->r[i];
        phi1 = fields_test_old->phi[i - 1];
        phi2 = fields_test_old->phi[i];
        phi3 = fields_test->phi[i - 1];

        r = (r2 + r3) / 2.0;
        r_uderiv = first_deriv_uo2(r1, r2, r3, r4);
        r_vderiv = first_deriv_vo2(r1, r2, r3, r4);

        phi = (phi2 + phi3) / 2.0;
        phi_uderiv = first_deriv_uo1(phi1, phi3);
        phi_vderiv = first_deriv_vo1(phi1, phi2);

        phi_dot = (r != 0 ? -1.0 / r * (r_uderiv * phi_vderiv + r_vderiv * phi_uderiv) : 0.0);

        phi4tilda = phi3 + phi2 - phi1 + du * dv * phi_dot;

        // corrector stage  
        phi_uderiv = first_deriv_uo2(phi1, phi2, phi3, phi4tilda);
        phi_vderiv = first_deriv_vo2(phi1, phi2, phi3, phi4tilda);

        phi_dot = (r != 0 ? -1.0 / r * (r_uderiv * phi_vderiv + r_vderiv * phi_uderiv) : 0.0);

        fields_test->phi[i] = phi3 + phi2 - phi1 + du * dv * phi_dot;
    }
    
    fields->sigma[0] = 0.0;
    for (int i = 1; i < v_size; ++i) {
        double sigma_dot;

        double sigma, sigma1, sigma2, sigma3;
        double phi, phi_uderiv, phi_vderiv;
        double r, r_uderiv, r_vderiv;
        double r1, r2, r3, r4;
        double phi1, phi2, phi3, phi4;

        sigma1 = fields_old->sigma[i - 1];
        sigma2 = fields_old->sigma[i];
        sigma3 = fields->sigma[i - 1];
        r1 = fields_old->r[i - 1];
        r2 = fields_old->r[i];
        r3 = fields->r[i - 1];
        r4 = fields->r[i];
        phi1 = fields_old->phi[i - 1];
        phi2 = fields_old->phi[i];
        phi3 = fields->phi[i - 1];
        phi4 = fields->phi[i];

        sigma = (sigma2 + sigma3) / 2.0;
        r = (r2 + r3) / 2.0;
        r_uderiv = first_deriv_uo2(r1, r2, r3, r4);
        r_vderiv = first_deriv_vo2(r1, r2, r3, r4);
        phi_uderiv = first_deriv_uo2(phi1, phi2, phi3, phi4);
        phi_vderiv = first_deriv_vo2(phi1, phi2, phi3, phi4);

        sigma_dot = (r != 0 ? 2.0 * r_uderiv * r_vderiv / (r * r) + exp(sigma) / (2.0 * r * r) : 0.0) - 2.0 * phi_uderiv * phi_vderiv;

        fields->sigma[i] = sigma3 + sigma2 - sigma1 + du * dv * sigma_dot;
    }
    init_fields->sigma_v0[n] = 0.0; 
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function to set the variables at u_max */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_fields_u_max(int n, Fields* fields, Fields* fields_u_max) {

    int u_loc = v_size-1;

    fields_u_max->phi[n]    = fields->phi[u_loc];
    fields_u_max->r[n]      = fields->r[u_loc];
    fields_u_max->sigma[n]  = fields->sigma[u_loc];

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function to save the correlator matrix at the end of the simulation */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_correlator(int n, Fields* fields, Q_Fields** q_fields, Q_Fields** q_fields_star, double** correlator_half) {

    for (int i = 0; i < v_size; ++i) {
        for (int j = 0; j < n; ++j) {

            __complex__ double Phi_mode1, Phi_mode2;
            double phi_phi, phi1_phi1, phi2_phi2;

            int l_value;
            phi_phi = 0;
            phi1_phi1 = 0;
            phi2_phi2 = 0;


            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                        l_value = l_start + l * l_step;

                        double r;
                        r = fields->r[i];

                        Phi_mode1 = pow(r, l_value) * (q_fields[which_q_field]->phi[k][l][i]);

                        Phi_mode2 = (q_fields_star[which_q_field]->phi[k][l][j]);


                        phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__(Phi_mode1 * conj(Phi_mode2));
                        
                    }
                }
            }
            correlator_half[i][j] = phi_phi;
        }
    }
    FILE* finout;
    finout = fopen("correlator_Ori.txt", "w");
    for (int i = 0; i < v_size; ++i) {
        fprintf(finout, "\n");
        for (int j = 0; j < n; ++j) {
            fprintf(finout, "%.20f ", (correlator_half[i][j]));
        }
    }
    fclose(finout);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Full evolution of all dynamical variables */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void full_evolution(double** full_r, Initial_Fields* init_fields, Fields* fields, Fields* fields_old, Initial_Fields* init_fields_test, 
                                           Fields* fields_test, Fields* fields_test_old, Initial_Q_Fields** init_q_fields, Q_Fields** q_fields, Q_Fields** q_fields_old, Fields* fields_u_max, Q_Fields** q_fields_star, double** correlator_half) {

    double horizonv[v_size];
    double horizonu[v_size];
    double constru[u_size];
    double constrv[v_size];

    double time[v_size];
    double vpoints[v_size];
    double upoints[v_size];


    for (int m = 0; m < v_size; ++m) {
        time[m] = (dv*m) / 2.0;
        vpoints[m] = dv * m;
        upoints[m] = 0.0;
    }

    double cosm_const = 0.0;
    cosm_const = find_phi_phi(0, 0, fields, q_fields);
    printf("Cosm constant = %.5f", cosm_const);

    set_fields_old(fields, fields_old, fields_test, fields_test_old, q_fields, q_fields_old);
    for (int n = 0; n < (u_size); ++n) {

        for (int k = 0; k < number_of_k_modes; ++k) {
            int l_value;
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;

                    q_fields_star[which_q_field]->phi[k][l][n] = pow(fields->r[v_size-1],l_value) * q_fields[which_q_field]->phi[k][l][v_size-1];
                }
            }
        }

        set_fields_old(fields, fields_old, fields_test, fields_test_old, q_fields, q_fields_old);

        for (int m = 0; m < v_size; ++m) {

            
            double phiphi = 0.0;
            phiphi = find_phi_phi(n, m, fields, q_fields);
            full_r[n][m] = fields->r[m];
        }
        vpoints[0] = 0.0;
        for (int m = 0; m < v_size; ++m) {
            time[m] = time[m] + exp(fields->sigma[m] / 2.0)*first_derivv(m, fields->r) * ( du);
            vpoints[m + 1] = vpoints[m] + exp(fields->sigma[m] / 2.0)*dv;
            upoints[m] = upoints[m] + exp(fields->sigma[m] / 2.0) * du;
        }

        if (n == 0) {
            check_constrsv(constrv, fields);
            save_constrv(constrv);
        }
        set_boundary_conds((int)(n + 1), init_fields, fields, init_fields_test, fields_test, init_q_fields, q_fields);

        diamond_evolution(n, init_fields, fields, fields_old, init_fields_test, fields_test, fields_test_old, init_q_fields, q_fields, q_fields_old);

        set_fields_u_max(n, fields, fields_u_max);

        if (n == 500) {
            save_correlator(n, fields, q_fields, q_fields_star, correlator_half);
        }
        
        horizonv[n] = 0.0;
        horizonu[n] = 0.0;
        for (int i = 0; i < v_size; ++i) {
            double r1, r2, r3, r4, r_vderiv;
            r1 = fields_old->r[i - 1];
            r2 = fields_old->r[i];
            r3 = fields->r[i - 1];
            r4 = fields->r[i];
            r_vderiv = first_derivv(i, fields->r);
            if (r_vderiv < 0) {
                printf("r' negative at r=%.5f\n", (r4) / 1.0);
                horizonv[i] = i*dv;
                horizonu[i] = n*du;
                break;
            }
        }
        printf("n=%d\n", n);

        
    }
    

    check_constrsu(constru, fields_u_max);
    save_constru(constru);

    save_horizonu(horizonu);
    save_horizonv(horizonv);


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Free memory that was allocated before for the fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void free_memory(Fields* fields, Q_Fields** q_fields) {
    free(fields->phi);
    free(fields->r);
    free(fields->sigma);
    free(fields);

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(q_fields[which_q_field]->phi[k][l]);


            }
            free(q_fields[which_q_field]->phi[k]);

        }
        free(q_fields[which_q_field]->phi);

        free(q_fields[which_q_field]);

    }
    free(q_fields);


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Free memory for the initial fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void free_memory_init(Initial_Fields* init_fields, Initial_Q_Fields** init_q_fields) {

    free(init_fields->phi_v0);
    free(init_fields->phi_u0);
    free(init_fields->r_v0);
    free(init_fields->r_u0);
    free(init_fields->sigma_v0);
    free(init_fields->sigma_u0);
    free(init_fields);


    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(init_q_fields[which_q_field]->phi_v0[k][l]);
                free(init_q_fields[which_q_field]->phi_u0[k][l]);

            }
            free(init_q_fields[which_q_field]->phi_v0[k]);
            free(init_q_fields[which_q_field]->phi_u0[k]);
        }
        free(init_q_fields[which_q_field]->phi_v0);
        free(init_q_fields[which_q_field]->phi_u0);


        free(init_q_fields[which_q_field]);

    }
    free(init_q_fields);

}

/* Main */
void main() {


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* DEFINE VARIABLES AND ASSIGN ALL THE MEMORY *//////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Fields*         fields;
    Fields*         fields_old;
    Fields*         fields_test;
    Fields*         fields_test_old;

    Fields*         fields_u_max;

    Q_Fields**      q_fields;
    Q_Fields**      q_fields_old;
    Q_Fields**      q_fields_star;


    Initial_Fields* init_fields;
    Initial_Fields* init_fields_test;

    Initial_Q_Fields** init_q_fields;


    fields = (Fields*)malloc(sizeof(Fields));
    fields->phi           = (double*)malloc(v_size * sizeof(double));
    fields->r             = (double*)malloc(v_size * sizeof(double));
    fields->sigma         = (double*)malloc(v_size * sizeof(double));

    fields_old = (Fields*)malloc(sizeof(Fields));
    fields_old->phi       = (double*)malloc(v_size * sizeof(double));
    fields_old->r         = (double*)malloc(v_size * sizeof(double));
    fields_old->sigma     = (double*)malloc(v_size * sizeof(double));

    fields_test = (Fields*)malloc(sizeof(Fields));
    fields_test->phi           = (double*)malloc(v_size * sizeof(double));
    fields_test->r             = (double*)malloc(v_size * sizeof(double));
    fields_test->sigma         = (double*)malloc(v_size * sizeof(double));

    fields_test_old = (Fields*)malloc(sizeof(Fields));
    fields_test_old->phi           = (double*)malloc(v_size * sizeof(double));
    fields_test_old->r             = (double*)malloc(v_size * sizeof(double));
    fields_test_old->sigma         = (double*)malloc(v_size * sizeof(double));

    fields_u_max = (Fields*)malloc(sizeof(Fields));
    fields_u_max->phi           = (double*)malloc(v_size * sizeof(double));
    fields_u_max->r             = (double*)malloc(v_size * sizeof(double));
    fields_u_max->sigma         = (double*)malloc(v_size * sizeof(double));

    init_fields_test = (Initial_Fields*)malloc(sizeof(Initial_Fields));
    init_fields_test->phi_u0   = (double*)malloc(u_size * sizeof(double));
    init_fields_test->phi_v0   = (double*)malloc(v_size * sizeof(double));
    init_fields_test->r_u0     = (double*)malloc(u_size * sizeof(double));
    init_fields_test->r_v0     = (double*)malloc(v_size * sizeof(double));
    init_fields_test->sigma_u0 = (double*)malloc(u_size * sizeof(double));
    init_fields_test->sigma_v0 = (double*)malloc(v_size * sizeof(double));

    init_fields = (Initial_Fields*)malloc(sizeof(Initial_Fields));
    init_fields->phi_u0   = (double*)malloc(u_size * sizeof(double));
    init_fields->phi_v0   = (double*)malloc(v_size * sizeof(double));
    init_fields->r_u0     = (double*)malloc(u_size * sizeof(double));
    init_fields->r_v0     = (double*)malloc(v_size * sizeof(double));
    init_fields->sigma_u0 = (double*)malloc(u_size * sizeof(double));
    init_fields->sigma_v0 = (double*)malloc(v_size * sizeof(double));


    q_fields      = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));
    q_fields_old  = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));
    q_fields_star = (Q_Fields**)malloc(number_of_q_fields * sizeof(Q_Fields*));

    init_q_fields = (Initial_Q_Fields**)malloc(number_of_q_fields * sizeof(Initial_Q_Fields));


    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {


        q_fields[which_q_field]          = (Q_Fields*)malloc(sizeof(Q_Fields));
        q_fields_old[which_q_field]      = (Q_Fields*)malloc(sizeof(Q_Fields));
        q_fields_star[which_q_field]     = (Q_Fields*)malloc(sizeof(Q_Fields));

        init_q_fields[which_q_field] = (Initial_Q_Fields*)malloc(sizeof(Initial_Q_Fields));

        q_fields[which_q_field]->phi         = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_old[which_q_field]->phi     = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_star[which_q_field]->phi    = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        init_q_fields[which_q_field]->phi_v0 = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        init_q_fields[which_q_field]->phi_u0 = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));


        for (int k = 0; k < number_of_k_modes; k++) {
            q_fields[which_q_field]->phi[k]         = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_old[which_q_field]->phi[k]     = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_star[which_q_field]->phi[k]    = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            init_q_fields[which_q_field]->phi_v0[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            init_q_fields[which_q_field]->phi_u0[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));


            for (int l = 0; l < number_of_l_modes; ++l) {
                q_fields[which_q_field]->phi[k][l]         = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_old[which_q_field]->phi[k][l]     = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));
                q_fields_star[which_q_field]->phi[k][l]    = (__complex__ double*)malloc(u_size * sizeof(__complex__ double));

                init_q_fields[which_q_field]->phi_v0[k][l] = (__complex__ double*)malloc(u_size * sizeof(__complex__ double));
                init_q_fields[which_q_field]->phi_u0[k][l] = (__complex__ double*)malloc(v_size * sizeof(__complex__ double));

            }
        }
    }

    double** full_r;
    full_r = (double**)malloc(u_size * sizeof(double*));
    for (int i = 0; i < u_size; i++) {
        full_r[i] = (double*)malloc(v_size * sizeof(double));
    }
    double** correlator_half;
    correlator_half = (double**)malloc(v_size * sizeof(double*));
    for (int i = 0; i < v_size; i++) {
        correlator_half[i] = (double*)malloc(u_size * sizeof(double));
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* ACTUAL EVOLUTION OF FIELDS *//////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* INITIAL CONDITIONS */
    double v[v_size];
    make_points(v);
    save_points(v);
    save_upoints();
    charac_initial_conditions(init_fields, init_fields_test);
    charac_initial_conditions_quantum(init_q_fields, init_fields);

    set_u0_slice(init_fields, fields, init_fields_test, fields_test, init_q_fields, q_fields);

    double cosm_const = 0.0;
    cosm_const = find_phi_phi(0, 0, fields, q_fields);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ACTUAL EVOLUTION */

    full_evolution(full_r, init_fields, fields, fields_old, init_fields_test, fields_test, fields_test_old, init_q_fields, q_fields, q_fields_old, fields_u_max, q_fields_star, correlator_half);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* SAVE FIELDS */////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    save_r(full_r);

    free_memory(fields, q_fields);

    free_memory(fields_old, q_fields_old);

    free_memory_init(init_fields, init_q_fields);



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* FREE UP ALL THE MEMORY *//////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    free(fields_u_max->phi);
    free(fields_u_max->r);
    free(fields_u_max->sigma);
    free(fields_u_max);

    free(fields_test->phi);
    free(fields_test->r);
    free(fields_test->sigma);
    free(fields_test);

    free(fields_test_old->phi);
    free(fields_test_old->r);
    free(fields_test_old->sigma);
    free(fields_test_old);

    free(init_fields_test->phi_v0);
    free(init_fields_test->phi_u0);
    free(init_fields_test->r_v0);
    free(init_fields_test->r_u0);
    free(init_fields_test->sigma_v0);
    free(init_fields_test->sigma_u0);
    free(init_fields_test);


    for (int i = 0; i < u_size; i++) {
        free(full_r[i]);
    }
    free(full_r);

    printf("done");
}
