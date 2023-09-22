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
#define         M_P                                1.0                                                                                  //sqrt(8*PI)
#define         c                                  1.0                                                                                  //speed of light

/* GRID PARAMETERS */
#define         lattice_size                       100
#define         L_0                                1.0
#define         dt                                 10.0 / ((lattice_size) * (lattice_size))                                             //dt=1/latt_size^2 to preserve accuracy of spectral differentiation

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
#define         mSqGhost                           1.0//1.0*MpSq;                                                           // base mass of the Pauli-Villars regulator fields
double          massSq[number_of_q_fields]       = { muSq, mSqGhost, 3.0 * mSqGhost, mSqGhost, 3.0 * mSqGhost, 4.0 * mSqGhost };         // masses of the ghost fields
double          ghost_or_physical[6]             = { 1 , -1 , 1 , -1 , 1, -1 };                                                          // distinguishing between the real and ghost fields

/* QUANTUM MODE PARAMETERS */
#define         k_min                              1.0*PI/15.0 //PI/6.0;//5.0*2.0*PI/(lattice_size*dr);                                  // minimum value of k, also =dk
#define         dk                                 k_min            
#define         number_of_k_modes                  1                                                                                     // number of k modes
#define         number_of_l_modes                  1                                                                                     // number of l modes
#define         k_start                            0
#define         l_start                            0                                                                                    //the range of l is l_start, l_start+l_step, l_start+2l_step...
#define         l_step                             1

/* SIMULATION PARAMETERS */
#define         step                               10
#define         evolve_time                        1
#define         evolve_time_int                    1*lattice_size*lattice_size/10

/* NUMERICAL TIME ITERATION PARAMETERS */
#define         nu_legendre                        5
#define         number_of_RK_implicit_iterations   10

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Creating a structure for the variables */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct classical_fields{
    double *pi;
    double *phi;
};
typedef struct classical_fields Classical_Fields;
struct metric_fields {
    double* A;
    double* B;
    double* D_B;
    double* U_tilda;
    double* K;
    double* K_B;
    double* lambda;
    double* alpha;
    double* D_alpha;
};
typedef struct metric_fields Metric_Fields;
struct quantum_fields {
    __complex__ double*** phi;
    __complex__ double*** pi;
};
typedef struct quantum_fields Quantum_Fields;
struct stress_tensor {
    double rho;
    double j_A;
    double S_A;
    double S_B;
};
typedef struct stress_tensor Stress_Tensor;
struct bi_linears {
    double phi_phi;
    double chi_chi;
    double pi_pi;
    double chi_pi;
    double del_theta_phi_del_theta_phi_over_r_sq;
};
typedef struct bi_linears Bi_Linears;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that makes shifted Chebyshev points so that r=[0,infinity]*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void make_points(double r[lattice_size]) {
    for (int i = 0; i < lattice_size; ++i) {
        r[i] = -cos((i+lattice_size) * PI / (2.0*lattice_size));
        r[i] = L_0*r[i] / sqrt(1.0 - r[i]*r[i]);
    }
    r[0] = 0.0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_c_i(double c_i[nu_legendre]){//these were calculated in Mathematics using, for example, N[Roots[LegendreP[6, x] == 0, x], 20]

    double zeros_of_P[nu_legendre];//={0.0};

    if(nu_legendre==2){
        zeros_of_P[0] = -sqrt(3.0)/3.0;
        zeros_of_P[1] =  sqrt(3.0)/3.0;
    }
    if(nu_legendre==3){
        zeros_of_P[0] = -sqrt(15.0)/5.0;
        zeros_of_P[1] =  0.0;
        zeros_of_P[2] =  sqrt(15.0)/5.0;
    }
    if(nu_legendre==4){
        zeros_of_P[0] = -sqrt(525.0+70.0*sqrt(30.0))/35.0;
        zeros_of_P[1] = -sqrt(525.0-70.0*sqrt(30.0))/35.0;
        zeros_of_P[2] =  sqrt(525.0-70.0*sqrt(30.0))/35.0;
        zeros_of_P[3] =  sqrt(525.0+70.0*sqrt(30.0))/35.0;
    }
    if(nu_legendre==5){
        zeros_of_P[0] = -sqrt(245.0+14.0*sqrt(70.0))/21.0;
        zeros_of_P[1] = -sqrt(245.0-14.0*sqrt(70.0))/21.0;
        zeros_of_P[2] =  0.0;
        zeros_of_P[3] =  sqrt(245.0-14.0*sqrt(70.0))/21.0;
        zeros_of_P[4] =  sqrt(245.0+14.0*sqrt(70.0))/21.0;
    }
    if(nu_legendre==6){
        zeros_of_P[0] = -0.93246951420315202781;
        zeros_of_P[1] = -0.66120938646626451366;
        zeros_of_P[2] = -0.23861918608319690863;
        zeros_of_P[3] =  0.23861918608319690863;
        zeros_of_P[4] =  0.66120938646626451366;
        zeros_of_P[5] =  0.93246951420315202781;
    }
    if(nu_legendre==7){
        zeros_of_P[0] = -0.94910791234275852453;
        zeros_of_P[1] = -0.74153118559939443986;
        zeros_of_P[2] = -0.40584515137739716691;
        zeros_of_P[3] =  0.0;
        zeros_of_P[4] =  0.40584515137739716691;
        zeros_of_P[5] =  0.74153118559939443986;
        zeros_of_P[6] =  0.94910791234275852453;
    }

    for(int i=0;i<nu_legendre;++i){
        c_i[i] = (zeros_of_P[i]+1.0)/2.0 ;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_a_ij__b_i(double c_i[nu_legendre], double b_i[nu_legendre], double a_ij[nu_legendre][nu_legendre], double GL_matrix_inverse[nu_legendre][nu_legendre]){
    double RHS_vector1[nu_legendre], RHS_vector2[nu_legendre];

    for(int row=0;row<nu_legendre;++row){
        for(int j=0;j<nu_legendre;++j){
            RHS_vector1[j] = pow(c_i[row],j+1)/(j+1);
            RHS_vector2[j] = 1.0/(j+1);
        }
        for(int i=0;i<nu_legendre;++i){
            a_ij[row][i]=0.0;
            for(int j=0;j<nu_legendre;++j){
                a_ij[row][i] = a_ij[row][i] + GL_matrix_inverse[i][j]*RHS_vector1[j];
            }
        }
    }

    for(int i=0;i<nu_legendre;++i){
        b_i[i] = 0.0;
        for(int j=0;j<nu_legendre;++j){
            b_i[i] = b_i[i] + GL_matrix_inverse[i][j]*RHS_vector2[j];
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-Gauss-Legendre matrix inverse---------------------
void find_GL_matrix_inverse(double c_i[nu_legendre], double GL_matrix_inverse[nu_legendre][nu_legendre]){
    double determinant, row_factor[nu_legendre], linear_sum[nu_legendre], quadratic_sum[nu_legendre], cubic_sum[nu_legendre], quartic_sum[nu_legendre];

    //first get the determinant
    determinant = 1.0;
    for(int i=0;i<nu_legendre;++i){
        for(int j=i+1;j<nu_legendre;++j){
            determinant = determinant*(c_i[j] - c_i[i]);
        }
    }

    //this gives determinants of  {(c1-c0); (c1-c0)(c2-c0)(c2-c1); (c1-c0)(c2-c0)(c3-c0)(c2-c1)(c3-c1)(c3-c2)}

    for(int row=0;row<nu_legendre;++row){
        row_factor[row]=1.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                if(i!=row && j!=row) row_factor[row] = row_factor[row]*(c_i[j]-c_i[i]);
            }
        }

        linear_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            if(i!=row)linear_sum[row] = linear_sum[row] + c_i[i];
        }

        quadratic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                if(i!=row && j!=row)quadratic_sum[row] = quadratic_sum[row] + c_i[i]*c_i[j];
            }
        }
        cubic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                for(int k=j+1;k<nu_legendre;++k){
                    if(i!=row && j!=row && k!=row)cubic_sum[row] = cubic_sum[row] + c_i[i]*c_i[j]*c_i[k];
                }
            }
        }
        quartic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for (int j = i + 1; j < nu_legendre; ++j) {
                for (int k = j + 1; k < nu_legendre; ++k) {
                    for (int l = k + 1; l < nu_legendre; ++l) {
                        if (i != row && j != row && k != row && l != row)quartic_sum[row] = quartic_sum[row] + c_i[i] * c_i[j] * c_i[k] * c_i[l];
                    }
                }
            }
        }

    }
    for (int col = 0; col < nu_legendre; ++col) {
        for (int row = 0; row < nu_legendre; ++row) {
            if (col == 0)GL_matrix_inverse[row][col] = row_factor[row] * quartic_sum[row] / determinant;
            if (col == 1)GL_matrix_inverse[row][col] = row_factor[row] * cubic_sum[row] / determinant;
            if (col == 2)GL_matrix_inverse[row][col] = row_factor[row] * quadratic_sum[row] / determinant;
            if (col == 3)GL_matrix_inverse[row][col] = row_factor[row] * linear_sum[row] / determinant;
            if (col == 4)GL_matrix_inverse[row][col] = row_factor[row] / determinant;
            GL_matrix_inverse[row][col] = pow(-1, (row + col)) * GL_matrix_inverse[row][col];
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dot products for matrix and vector calculations*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dot_product_for_phys(double** matrix, double* field, double* product) {
    if (matrix[0][0] == 1.0) {
        for (int i = 0; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 0; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i][j] * field[j];
            }
        }
    }
    else {
        for (int i = 1; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 1; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i-1][j-1] * field[j];
            }
        }
        product[0] = 0.0;
    }
}
void dot_product_for_coeff(double** matrix, double* field) {
    double product[lattice_size];
    if (matrix[lattice_size-1][lattice_size-1] != 0.0) {
        for (int i = 0; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 0; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i][j] * field[j];
            }
        }

        for (int i = 0; i < lattice_size; i++) {
            field[i] = product[i];
        }
    }
    else {
        for (int i = 1; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 1; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i-1][j-1] * field[j];
            }
        }
        product[0] = 0.0;


        for (int i = 0; i < lattice_size; i++) {
            field[i] = product[i];
        }
    }

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Complex version of dot products for matrix and vector calculations*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dot_product_for_phys_comp(double** matrix, __complex__ double* field, __complex__ double* product) {
    if (matrix[0][0] == 1.0) {
        for (int i = 0; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 0; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i][j] * field[j];
            }
        }
    }
    else {
        for (int i = 1; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 1; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i - 1][j - 1] * field[j];
            }
        }
        product[0] = 0.0;
    }
}
void dot_product_for_coeff_comp(double** matrix, __complex__ double* field) {
    __complex__ double product[lattice_size];
    if (matrix[lattice_size - 1][lattice_size - 1] != 0.0) {
        for (int i = 0; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 0; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i][j] * field[j];
            }
        }

        for (int i = 0; i < lattice_size; i++) {
            field[i] = product[i];
        }
    }
    else {
        for (int i = 1; i < lattice_size; i++) {
            product[i] = 0.0;
            for (int j = 1; j < lattice_size; j++) {
                product[i] = product[i] + matrix[i - 1][j - 1] * field[j];
            }
        }
        product[0] = 0.0;


        for (int i = 0; i < lattice_size; i++) {
            field[i] = product[i];
        }
    }

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* This function provides a version of gsl's Bessel function that ignores any underflow error */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* This function provides the initial profile functions */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double phi_mode_profile_0(double k, int l, double r) {
    return (sqrt(k / PI) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l));
}
double phi_mode_profile_0_prime(double k, int l, double r) {
    return (-k * sqrt(k / PI) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l));
}
double phi_mode_profile_massive(double msq, double k, int l, double r) {
    return (k / sqrt(PI * sqrt(k * k + msq)) * gsl_sf_bessel_jl_safe(l, k * r) / pow(r, l));
}
double phi_mode_profile_massive_prime(double msq, double k, int l, double r) {
    return (-k * k / sqrt(PI * sqrt(k * k + msq)) * gsl_sf_bessel_jl_safe(l + 1, k * r) / pow(r, l));
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Calculating initial A(r) */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double dA_dr(double r, double A_current) {
    double chi;

    //chi = ( r != 0.0 ? (-k_val * sqrt(k_val / PI) * gsl_sf_bessel_jl_safe(l_val + 1, k_val * r) / pow(r, l_val)) : 0.0);

    chi = - amplitude * 2.0 * r / (initial_width * initial_width) * exp(-r * r / (initial_width * initial_width));
        //2.0 * amplitude * r / (initial_width * initial_width) * exp(-(pow(((r - initial_radius) / initial_width), 2))) * (1.0 - r * (r - initial_radius) / pow(initial_width, 2))
        //+ 2.0 * amplitude * r / (initial_width * initial_width) * exp(-(pow(((r + initial_radius) / initial_width), 2))) * (1.0 - r * (r + initial_radius) / pow(initial_width, 2));

    return A_current * (1.0 / r * (1.0 - A_current) + r / (2.0 * M_P * M_P) * chi * chi);
}
int initialise_A(double c_i[nu_legendre], double b_i[nu_legendre], double a_ij[nu_legendre][nu_legendre], double A_RK1, double A_RK2, double A_RK3,
    double A_RK4, double A_RK5, double A_RK_sum, double A, double A_init[lattice_size]) {

    double rpoints[lattice_size];
    double dr, r;
    make_points(rpoints);

    for (int i = 0; i < lattice_size - 1; i++) {
        r  = rpoints[i];
        dr = rpoints[i + 1] - rpoints[i];

        A_RK1 = 0;
        A_RK2 = 0;
        A_RK3 = 0;
        A_RK4 = 0;
        A_RK5 = 0;

        //do a few RK iterations in order to converge on the implicit solution
        for (int iter = 0; iter < number_of_RK_implicit_iterations; ++iter) {
            //iterate the RK1 term
            A_RK_sum = A + dr * (a_ij[0][0] * A_RK1 + a_ij[0][1] * A_RK2 + a_ij[0][2] * A_RK3 + a_ij[0][3] * A_RK4 + a_ij[0][4] * A_RK5);

            A_RK1 = dA_dr(r + c_i[0] * dr, A_RK_sum);

            //then iterate the RK2 term
            A_RK_sum = A + dr * (a_ij[1][0] * A_RK1 + a_ij[1][1] * A_RK2 + a_ij[1][2] * A_RK3 + a_ij[1][3] * A_RK4 + a_ij[1][4] * A_RK5);

            A_RK2 = dA_dr(r + c_i[1] * dr, A_RK_sum);

            //then iterate the RK3 term
            A_RK_sum = A + dr * (a_ij[2][0] * A_RK1 + a_ij[2][1] * A_RK2 + a_ij[2][2] * A_RK3 + a_ij[2][3] * A_RK4 + a_ij[2][4] * A_RK5);

            A_RK3 = dA_dr(r + c_i[2] * dr, A_RK_sum);

            //then iterate the RK4 term
            A_RK_sum = A + dr * (a_ij[3][0] * A_RK1 + a_ij[3][1] * A_RK2 + a_ij[3][2] * A_RK3 + a_ij[3][3] * A_RK4 + a_ij[3][4] * A_RK5);

            A_RK4 = dA_dr(r + c_i[3] * dr, A_RK_sum);

            //then iterate the RK5 term
            A_RK_sum = A + dr * (a_ij[4][0] * A_RK1 + a_ij[4][1] * A_RK2 + a_ij[4][2] * A_RK3 + a_ij[4][3] * A_RK4 + a_ij[4][4] * A_RK5);

            A_RK5 = dA_dr(r + c_i[4] * dr, A_RK_sum);
        }
        //add up the RK contributions

        A = A + dr * (b_i[0] * A_RK1 + b_i[1] * A_RK2 + b_i[2] * A_RK3 + b_i[3] * A_RK4 + b_i[4] * A_RK5);

        A_init[i+1] = A;
        //printf("\n%.60f, ", A);
    }

}
void calc_A(double A_initial[lattice_size]) {
    
    double rpoints[lattice_size];

    double   a_ij[nu_legendre][nu_legendre];             //coefficients of the Runge-Kutta evolution
    double   b_i[nu_legendre];                           //coefficients of the Runge-Kutta evolution
    double   c_i[nu_legendre];                           //the zeros of P_nu(2c - 1)=0, i.e. P_nu(2c_1 - 1)
    double   GL_matrix_inverse[nu_legendre][nu_legendre];         //this comes from GL_matrix*(a_ij)=(c_i^l) for (a_ij)
                                                                                     //                          (b_i)  (1/l  )     (b_i )
    double A, A_RK1, A_RK2, A_RK3, A_RK4, A_RK5, A_RK_sum;

    make_points(rpoints);  // make lattice points

    A_initial[0] = 1.0;
    A = A_initial[0];

    find_c_i(c_i);
    find_GL_matrix_inverse(c_i, GL_matrix_inverse);
    find_a_ij__b_i(c_i, b_i, a_ij, GL_matrix_inverse);

    initialise_A(c_i, b_i, a_ij, A_RK1, A_RK2, A_RK3, A_RK4, A_RK5, A_RK_sum, A, A_initial);

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initialise fields and metric functions */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initialise(Classical_Fields* fieldd, Metric_Fields* metric, double** B, double** B_as, double** B_inv_sym, double** B_inv_asym) {
    double r[lattice_size];
    make_points(r);
    double chi[lattice_size];
    double A_new[lattice_size];
    double U_new[lattice_size];
    double lamb_new[lattice_size];

    calc_A(metric->A);
    
    for (int i = 1; i < lattice_size; ++i) {
        chi[i] = - amplitude * 2.0 * r[i] / (initial_width * initial_width) * exp(-r[i] * r[i] / (initial_width * initial_width));
                //2.0 * amplitude * r[i] / (initial_width * initial_width) * exp(-(pow(((r[i] - initial_radius) / initial_width), 2))) * (1.0 - r[i] * (r[i] - initial_radius) / pow(initial_width, 2))
               //+ 2.0 * amplitude * r[i] / (initial_width * initial_width) * exp(-(pow(((r[i] + initial_radius) / initial_width), 2))) * (1.0 - r[i] * (r[i] + initial_radius) / pow(initial_width, 2));


        metric->lambda[i]  = 1.0 / r[i] * (1.0 - metric->A[i] / metric->B[i]);
        metric->U_tilda[i] = 1.0 / r[i] * (1.0 - metric->A[i]) * (1.0 - 4.0 / metric->A[i]) + r[i] / (2.0 * M_P * M_P) * chi[i] * chi[i];

    }
    metric->lambda[0] = 0.0;
    metric->U_tilda[0] = 0.0;
    
    
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the quantum variables */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initial_conditions_quantum(Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric) {

    /* QUANTUM MATTER FIELDS */
    double rpoints[lattice_size];
    make_points(rpoints);
    //the initial data for the quantum vacuum modes phi
    //#pragma omp parallel for //num_threads(lattice_size)
    for (int i = 0; i < lattice_size; ++i) {
        //printf("Thread rank: %d\n", omp_get_thread_num());
        //static extern int GetCurrentProcessorNumber();
        //int myProcessorNum = GetCurrentProcessorNumber();

        //printf("Processor: %d\n", myProcessorNum);
        double k_wavenumber, omega_phi;
        int l_value;
        double r;
        r = rpoints[i];
        //#pragma omp parallel for
        for (int k = 0; k < number_of_k_modes; ++k) {
            k_wavenumber = (k_start + (k + 1)) * k_min;
            for (int l = 0; l < number_of_l_modes; ++l) {
                l_value = l_start + l * l_step;
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {//cycle through the quantum fields and initialize them
                    omega_phi = sqrt(k_wavenumber * k_wavenumber + massSq[which_q_field]);
                    if (massSq[which_q_field] == 0) {
                        if (i > 0) {
                            q_fields[which_q_field]->phi[k][l][i] = phi_mode_profile_0(k_wavenumber, l_value, r);                 //set the r!=0 zero values
                        }
                        else {
                            if (2 * l_value + 1 < GSL_SF_DOUBLEFACT_NMAX) {       //check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->phi[k][l][i] = sqrt(k_wavenumber / PI) * pow(k_wavenumber, l_value) / gsl_sf_doublefact(2 * l_value + 1);
                            }
                            else {
                                q_fields[which_q_field]->phi[k][l][i] = 0.0;
                            }
                        }
                    }
                    else {
                        if (i > 0) {
                            q_fields[which_q_field]->phi[k][l][i] = phi_mode_profile_massive(massSq[which_q_field], k_wavenumber, l_value, r);
                        }
                        else {                                                                                                       //this is the value at the origin
                            if (2 * l_value + 1 < GSL_SF_DOUBLEFACT_NMAX) {//check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->phi[k][l][i] = k_wavenumber / sqrt(PI * omega_phi) * pow(k_wavenumber, l_value) / gsl_sf_doublefact(2 * l_value + 1);
                            }
                            else {
                                q_fields[which_q_field]->phi[k][l][i] = 0.0;
                            }
                        }
                    }

                    //then sort out the momenta
                    q_fields[which_q_field]->pi[k][l][i] = -I * omega_phi * q_fields[which_q_field]->phi[k][l][i];                 //note that this is a specification of pi, and not phi_dot
                }

            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the classical variables */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initial_conditions_classical(Classical_Fields* fields, Metric_Fields* metric) {
    double r[lattice_size];
    make_points(r);

    for (int i = 0; i < lattice_size; ++i) {
        metric->A[i]       = 1.0;
        metric->B[i]       = 1.0;
        metric->D_B[i]     = 0.0;
        metric->K[i]       = 0.0;
        metric->K_B[i]     = 0.0;
        metric->alpha[i]   = 1.0;
        metric->D_alpha[i] = 0.0;
        metric->lambda[i]  = 0.0;
        metric->U_tilda[i] = 0.0;
    }

    for (int i = 0; i < lattice_size; ++i) {
        /* PHI */
        fields->phi[i] = amplitude * exp(-r[i] * r[i] / (initial_width * initial_width));
            //exp(-(r[i] * r[i]));
            //(amplitude * (r[i] / initial_width) * (r[i] / initial_width) * exp(-(r[i] - initial_radius) * (r[i] - initial_radius) / (initial_width * initial_width))) +
            //(amplitude * (r[i] / initial_width) * (r[i] / initial_width) * exp(-(-r[i] - initial_radius) * (-r[i] - initial_radius) / (initial_width * initial_width)));


        /* PI */
        fields->pi[i] = 0.0;

    }
    //fieldd->phi[0] = 0.0;
    //fieldd->pi[0] = 0.0;

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Calculate norm */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double norm(__complex__ double number) {
    double nor = 0.0;
    nor = (pow((__real__ number), 2.0) + pow((__imag__ number), 2.0));
    return nor;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Chebyshev polynomial definitions and their derivatives */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double R_k(int k, double r) {
    double R_nm1, R_n, R_np1;

    R_nm1 = 1.0/sqrt(1+r*r/(L_0*L_0));
    R_n = 2.0*r/L_0 * 1.0 /(1 + r * r / (L_0 * L_0));

    if (k != 0 && k != 1) {
        for (int i = 1; i < k; i++) {
            R_np1 = 2.0*r/L_0 * 1.0 /sqrt(1 + r * r / (L_0 * L_0)) * R_n - R_nm1;

            R_nm1 = R_n;
            R_n = R_np1;
        }
        return R_np1;
    }
    else if (k == 0) {
        return 1.0 / sqrt(1 + r * r / (L_0 * L_0));
    }
    else if (k == 1) {
        return 2.0 * r / L_0 * 1.0 / (1 + r * r / (L_0 * L_0));
    }
}
double R_k_sym(int k, double r) {
    double R;

    R = R_k(2*k, r);

    return R;
}
double R_k_asym(int k, double r) {
    double R;

    R = R_k(2*k+1, r);

    return R;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv_R_k(int k, double r) {

    double deriv = 0.0;
    double rho,rho_prime;
    double dR_n, dR_np1, dR_nm1;

    rho       = 2.0 * r / L_0 * 1.0 / sqrt(1 + r * r / (L_0 * L_0));
    rho_prime = 2.0*L_0/((L_0*L_0 + r*r)* sqrt(1 + r * r / (L_0 * L_0)));

    dR_nm1 = -r / (L_0 * L_0 * sqrt((1 + r * r / (L_0 * L_0))) * (1 + r * r / (L_0 * L_0)));
    dR_n   = (2.0 * L_0 * L_0 * L_0 - 2.0 * L_0 * r * r) / ((L_0 * L_0 + r * r) * (L_0 * L_0 + r * r));

    if (k != 0 && k != 1) {
        for (int i = 1; i < k; i++) {
            dR_np1 = rho * dR_n + rho_prime * R_k(i, r) - dR_nm1;

            dR_nm1 =dR_n;
            dR_n = dR_np1;
        }
        return dR_np1;
    }
    else if (k == 0) {
        return -r / (L_0*L_0*sqrt((1 + r * r / (L_0 * L_0)))* (1 + r * r / (L_0 * L_0)) );
    }
    else if (k == 1) {
        return (2.0* L_0*L_0*L_0 -2.0*L_0* r*r) / ( (L_0*L_0+r*r)*(L_0*L_0+r*r) );
    }

     
}
double first_deriv_R_k_sym(int k, double r) {
    double deriv = 0.0;

    deriv = first_deriv_R_k(2 * k, r);

    return deriv;
}
double first_deriv_R_k_asym(int k, double r) {
    double deriv = 0.0;

    deriv =first_deriv_R_k(2*k+1, r);

    return deriv;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Matrices made out of the Chebyshev polynomials */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_B_sym(double** B_phi) {
    double rpoints[lattice_size];
    make_points(rpoints);
    double r1;
    for (int i = 0; i < lattice_size; i++) {
        r1 = rpoints[i];
        for (int j = 0; j < lattice_size; j++) {
            B_phi[i][j] = R_k_sym(j, r1);
        }
    }
    
}
void find_B_asym(double** B_phi) {
    double rpoints[lattice_size];
    make_points(rpoints);
    double r1;
    for (int i = 1; i < lattice_size; i++) {
        r1 = rpoints[i];
        for (int j = 0; j < lattice_size-1; j++) {
            B_phi[i-1][j] = R_k_asym(j, r1);
        }
    }
    B_phi[lattice_size - 1][lattice_size - 1] = 0.0;

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Second derivatives for the Chebyshev polynomials */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double second_deriv_R_k(int k, double r) {
    
    double deriv = 0.0;
    double rho, rho_prime, rho_prime_prime;
    double ddR_n, ddR_np1, ddR_nm1;

    rho             = 2.0 * r / L_0 * 1.0 / sqrt(1 + r * r / (L_0 * L_0));
    rho_prime       = 2.0 * L_0 / ((L_0 * L_0 + r * r) * sqrt(1 + r * r / (L_0 * L_0)));
    rho_prime_prime = -6.0 * L_0*L_0*r / ((L_0 * L_0 + r * r) * (L_0 * L_0 + r * r)* sqrt(L_0 * L_0 + r * r));

    ddR_nm1 = L_0 * (2.0 * r * r - L_0 * L_0) / ((L_0 * L_0 + r * r) * (L_0 * L_0 + r * r) * sqrt(L_0 * L_0 + r * r));
    ddR_n   = (4.0 * L_0 * r * r * r - 12.0 * L_0 * L_0 * L_0 * r) / (((L_0 * L_0) + r * r) * ((L_0 * L_0) + r * r) * ((L_0 * L_0) + r * r));

    if (k != 0 && k != 1) {
        for (int i = 1; i < k; i++) {
            ddR_np1 = 2.0*rho_prime * first_deriv_R_k(i, r) + rho*ddR_n + rho_prime_prime * R_k(i, r) - ddR_nm1;
            ddR_nm1 = ddR_n;
            ddR_n = ddR_np1;
        }
        return ddR_np1;
    }
    else if (k == 0) {
        return L_0*(2.0*r*r - L_0*L_0)/ ((L_0 * L_0 + r * r) * (L_0 * L_0 + r * r) * sqrt(L_0 * L_0+ r * r));
    }
    else if (k == 1) {
        return (4.0 *L_0 * r*r*r - 12.0 * L_0*L_0*L_0 * r) / (((L_0 * L_0) + r * r) * ((L_0 * L_0) + r * r)* ((L_0 * L_0) + r * r));
    }
}
double second_deriv_R_k_sym(int k, double r) {
    double deriv = 0.0;

    deriv = second_deriv_R_k(2 * k, r);

    return deriv;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions for solving matrix equations - inverting matrices */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int  LUPDecompose(double** A, int N, double Tol, int* P) {

    int i, j, k, imax;
    double maxA, * ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; //failure, matrix is degenerate

        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            //pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            //counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1;  //decomposition done 
}
void LUPInvert(double** A, int* P, int N, double** IA) {

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j ? 1.0 : 0.0);

                for (int k = 0; k < i; k++)
                    IA[i][j] -= A[i][k] * IA[k][j];
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA[i][j] -= A[i][k] * IA[k][j];

            IA[i][j] /= A[i][i];
        }
    }
}
void invert_matrix(double** matrix, double** inverse) {
    int P[lattice_size +1];
    LUPDecompose(matrix, lattice_size, 0.001, P);
    LUPInvert(matrix, P, lattice_size, inverse);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Finding the inverses of the Chebyshev matrices */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_B_inverse_sym(double** B, double** B_inv) {
    
    invert_matrix(B, B_inv);
}
void find_B_inverse_asym(double** B, double** B_inv) {
    
    int P[lattice_size];
    LUPDecompose(B, lattice_size-1, 0.001, P);
    LUPInvert(B, P, lattice_size-1, B_inv);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions to calculate derivatives */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv_sym(int i, double** C_sym, double* field) {
    double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 0; j1 < lattice_size; ++j1) {
        field_prime = field_prime + C_sym[i][j1] * field[j1];
    }
    return field_prime;
    
}
double first_deriv_asym(int i, double** C_asym, double* field) {
    double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 1; j1 < lattice_size; ++j1) {
        field_prime = field_prime + C_asym[i][j1 - 1] * field[j1];
    }
    return field_prime;
}
double second_deriv(int i, double** D, double* field) {
    double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 0; j1 < lattice_size; ++j1) {
        field_prime = field_prime + D[i][j1] * field[j1];
    }
    return field_prime;
}
double field_phys_i_sym(int i, double** matrix, double* field) {
    double product = 0.0;
    for (int j = 0; j < lattice_size; j++) {
        product = product + matrix[i][j] * field[j];
    }
    return product;
}
double field_phys_i_asym(int i, double** matrix, double* field) {
    double product = 0.0;
    if (i == 0) {
        return 0.0;
    }
    else {
        for (int j = 1; j < lattice_size; j++) {
            product = product + matrix[i-1][j-1] * field[j];
        }
        return product;
    }   
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions to calculate complex derivatives */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double first_deriv_sym_comp(int i, double** C_sym, __complex__ double* field) {
    __complex__ double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 0; j1 < lattice_size/5; ++j1) {
        field_prime = field_prime + C_sym[i][j1] * field[j1];
    }
    return field_prime;
}
__complex__ double first_deriv_asym_comp(int i, double** C_asym, __complex__ double* field) {
    __complex__ double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 1; j1 < lattice_size/5; ++j1) {
        field_prime = field_prime + C_asym[i][j1 - 1] * field[j1];
    }
    return field_prime;
}
__complex__ double second_deriv_comp(int i, double** D, __complex__ double* field) {
    __complex__ double field_prime = 0.0;

    field_prime = 0.0;
    for (int j1 = 0; j1 < lattice_size/5; ++j1) {
        field_prime = field_prime + D[i][j1] * field[j1];
    }
    return field_prime;
}
__complex__ double field_phys_i_sym_comp(int i, double** matrix, __complex__ double* field) {
    __complex__ double product = 0.0;
    for (int j = 0; j < lattice_size; j++) {
        product = product + matrix[i][j] * field[j];
    }
    return product;
}
__complex__ double field_phys_i_asym_comp(int i, double** matrix, __complex__ double* field) {
    __complex__ double product = 0.0;
    if (i == 0) {
        return 0.0;
    }
    else {
        for (int j = 1; j < lattice_size; j++) {
            product = product + matrix[i - 1][j - 1] * field[j];
        }
        return product;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions to initialise matrices used in calculations */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_C_sym(double** C) {
    double rpoints[lattice_size];
    make_points(rpoints);
    double r1;
    for (int i = 0; i < lattice_size; i++) {
        r1 = rpoints[i];
        for (int j = 0; j < lattice_size; j++) {
            C[i][j] = first_deriv_R_k_sym(j, r1);
        }
    }
}
void find_C_asym(double** C) {
    double rpoints[lattice_size];
    make_points(rpoints);
    double r1;
    for (int i = 0; i < lattice_size; i++) {
        r1 = rpoints[i];
        for (int j = 0; j < lattice_size; j++) {
            C[i][j] = first_deriv_R_k_asym(j, r1);
        }
    }
}
void find_D_sym(double** D) {
    double rpoints[lattice_size];
    make_points(rpoints);
    double r1;
    for (int i = 0; i < lattice_size; i++) {
        r1 = rpoints[i];
        for (int j = 0; j < lattice_size; j++) {
            D[i][j] = second_deriv_R_k_sym(j, r1);
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the bilinears */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears(int i, Bi_Linears* bi_linears, Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric, double** B, double** C) {
    double rpoints[lattice_size];
    double r, r_l;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    int l_value;
    make_points(rpoints);
    r = rpoints[i];

    phi_phi = 0.0;
    chi_chi = 0.0;
    pi_pi   = 0.0;
    chi_pi  = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if (coherent_state_switch != 0) {
        phi_phi = field_phys_i_sym(i, B, c_fields->phi) * field_phys_i_sym(i, B, c_fields->phi);
        chi_chi = first_deriv_sym (i, C, c_fields->phi) * first_deriv_sym (i, C, c_fields->phi);
        pi_pi   = field_phys_i_sym(i, B, c_fields->pi)  * field_phys_i_sym(i, B, c_fields->pi);
        chi_pi  = first_deriv_sym (i, C, c_fields->phi) * field_phys_i_sym(i, B, c_fields->pi);
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
                    Phi_mode = r_l * field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]);

                    if (i == 0) {
                        Phi_mode_plus = pow(rpoints[i+1], l_value) * field_phys_i_sym_comp(i+1, B, q_fields[which_q_field]->phi[k][l]);
                    }

                    /* CHI MODE */
                    if (l_value == 0) {
                        Chi_mode = first_deriv_sym_comp(i, C, q_fields[which_q_field]->phi[k][l]);
                    }
                    else if (l_value == 1) {
                        Chi_mode = field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]) + r * first_deriv_sym_comp(i, C, q_fields[which_q_field]->phi[k][l]);
                    }
                    else {
                        Chi_mode = l_value * pow(r, l_value - 1) * field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]) + r_l * first_deriv_sym_comp(i, C, q_fields[which_q_field]->phi[k][l]);
                    }

                    /* PI MODE */

                    Pi_mode = r_l * field_phys_i_sym_comp(i, B, q_fields[which_q_field]->pi[k][l]);


                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Chi_mode);
                    pi_pi   = pi_pi   + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    chi_pi  = chi_pi  + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * (__real__(Pi_mode * conj(Chi_mode)));

                    if (i != 0) {
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);
                    }
                    else {//use the data at r=dr to estimate the r=0 case. This is only relevant for l=1
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode_plus) / (rpoints[1]* rpoints[1]);
                    }


                }
            }
        }
    }
    //printf("\n %.100f, ", norm(chi_mode));
    bi_linears->phi_phi = phi_phi;
    bi_linears->chi_chi = chi_chi;
    bi_linears->pi_pi = pi_pi;
    bi_linears->chi_pi = chi_pi;
    bi_linears->del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the biliears in the midpoints of the iteration */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears_midpoint(int i, Bi_Linears* bi_binears_midpoint, Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric, double** B, double** C){
    double rpoints[lattice_size];
    double r, r_l;
    double phi_phi = 0.0, psi_psi = 0.0, pi_pi = 0.0, psi_pi = 0.0, del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    __complex__ double Phi_mode, Psi_mode, Pi_mode;
    int l_value;

    make_points(rpoints);
    r = rpoints[i] + 0.5 * (rpoints[i+1]-rpoints[i]);
    phi_phi = 0.0;
    psi_psi = 0.0;
    pi_pi = 0.0;
    psi_pi = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if (coherent_state_switch != 0) {
        phi_phi = 0.25 * (field_phys_i_sym(i, B, c_fields->phi) + field_phys_i_sym(i+1, B, c_fields->phi)) * (field_phys_i_sym(i, B, c_fields->phi) + field_phys_i_sym(i+1, B, c_fields->phi));
        psi_psi = 0.25 * (first_deriv_sym (i, C, c_fields->phi) + first_deriv_sym (i+1, C, c_fields->phi)) * (first_deriv_sym (i, C, c_fields->phi) + first_deriv_sym (i+1, C, c_fields->phi));
        pi_pi   = 0.25 * (field_phys_i_sym(i, B, c_fields->pi)  + field_phys_i_sym(i+1, B, c_fields->pi))  * (field_phys_i_sym(i, B, c_fields->pi)  + field_phys_i_sym(i+1, B, c_fields->pi));
        psi_pi  = 0.25 * (first_deriv_sym (i, C, c_fields->phi) + first_deriv_sym (i+1, C, c_fields->phi)) * (field_phys_i_sym(i, B, c_fields->pi)  + field_phys_i_sym(i+1, B, c_fields->pi));
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    }

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = r^l pi
    //Psi = lr^{l-1} u + r^l psi
    if (hbar != 0) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    l_value = l_start + l * l_step;
                    r_l = pow(r, l_value);

                    Phi_mode = 0.5 * r_l * (field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]) + field_phys_i_sym_comp(i+1, B, q_fields[which_q_field]->phi[k][l]));

                    if (l_value == 0) {
                        Psi_mode = 0.5 * (first_deriv_sym_comp(i, C, q_fields[which_q_field]->phi[k][l]) + first_deriv_sym_comp(i+1, C, q_fields[which_q_field]->phi[k][l]));
                    }
                    else if (l_value == 1) {
                        Psi_mode = 0.5 * (field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]) + field_phys_i_sym_comp(i+1, B, q_fields[which_q_field]->phi[k][l]))
                                 + 0.5*r*(first_deriv_sym_comp (i, C, q_fields[which_q_field]->phi[k][l]) + first_deriv_sym_comp (i+1, C, q_fields[which_q_field]->phi[k][l]));
                    }
                    else {
                        Psi_mode = 0.5 * l_value * pow(r, l_value - 1) * (field_phys_i_sym_comp(i, B, q_fields[which_q_field]->phi[k][l]) + field_phys_i_sym_comp(i+1, B, q_fields[which_q_field]->phi[k][l]))
                            + 0.5 * r_l * (first_deriv_sym_comp(i, C, q_fields[which_q_field]->phi[k][l]) + first_deriv_sym_comp(i+1, C, q_fields[which_q_field]->phi[k][l]));
                    }

                    Pi_mode = 0.5 * r_l * (field_phys_i_sym_comp(i, B, q_fields[which_q_field]->pi[k][l]) + field_phys_i_sym_comp(i + 1, B, q_fields[which_q_field]->pi[k][l]));



                    phi_phi = phi_phi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Phi_mode);
                    psi_psi = psi_psi + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Psi_mode);
                    pi_pi   = pi_pi   + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * norm(Pi_mode);
                    psi_pi  = psi_pi  + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * (__real__(Pi_mode * conj(Psi_mode)));
                    del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * 0.5 * l_value * (l_value + 1.0) * (2.0 * l_value + 1.0) * norm(Phi_mode) / (r * r);

                }
            }
        }
    }

    bi_binears_midpoint->phi_phi = phi_phi;
    bi_binears_midpoint->chi_chi = psi_psi;
    bi_binears_midpoint->pi_pi = pi_pi;
    bi_binears_midpoint->chi_pi = psi_pi;
    bi_binears_midpoint->del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq;


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the cosmological constant */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double set_cosm_constant(Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric, double** Bmat, double** C) {
    double rho, S_A, A, B;
    int i;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    Bi_Linears    bi_linears;

    A = 1.0;
    B = 1.0;

    i = 0   ;


    set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, Bmat, C);

    phi_phi = bi_linears.phi_phi;
    chi_chi = bi_linears.chi_chi;
    pi_pi = bi_linears.pi_pi;
    chi_pi = bi_linears.chi_pi;
    del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

    rho = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq;
    S_A = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq;


    return rho;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Setting the stress tensor components for given variable fields */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_stress_tensor(int i, Stress_Tensor* stress_tnsr, Bi_Linears bi_linears, Metric_Fields* metric, Classical_Fields* c_fields, Quantum_Fields** q_fields, double** Bmat) {

    double rho = 0.0, j_A = 0.0, S_A = 0.0, S_B = 0.0;
    double A, B;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;

    rho = stress_tnsr->rho;
    j_A = stress_tnsr->j_A;
    S_A = stress_tnsr->S_A;
    S_B = stress_tnsr->S_B;

    A = field_phys_i_sym(i, Bmat, metric->A)+1.0;
    B = field_phys_i_sym(i, Bmat, metric->B)+1.0;


    phi_phi = bi_linears.phi_phi;
    chi_chi = bi_linears.chi_chi;
    pi_pi = bi_linears.pi_pi;
    chi_pi = bi_linears.chi_pi;
    del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

    rho = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / (B)*del_theta_phi_del_theta_phi_over_r_sq;
    j_A = -chi_pi / (sqrt(A) * B);
    S_A = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / (B)*del_theta_phi_del_theta_phi_over_r_sq;
    S_B = 1.0 / (2.0 * A) * (pi_pi / (B * B) - chi_chi);

    stress_tnsr->rho = rho;
    stress_tnsr->j_A = j_A;
    stress_tnsr->S_A = S_A;
    stress_tnsr->S_B = S_B;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Finding the right hand side of the time iteration equations (d_t f = ...) */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_f(Classical_Fields* c_fields, Classical_Fields* c_fields_dot, Quantum_Fields** q_fields, Quantum_Fields** q_fields_dot, Metric_Fields* metric, Metric_Fields* metric_dot, 
                                                      double** Bmat, double** Bmatas, double** B_inv_sym, double** B_inv_asym, double** C_sym, double** C_asym, double** D, double cosm_const) {
    

    double K_B_alpha[lattice_size], alpha_K[lattice_size], alpha_B_over_A[lattice_size], alpha_BA_chi[lattice_size];
    //#pragma omp parallel for 
    for (int m = 0; m < lattice_size; m++) {
        double r, rpoints[lattice_size];
        make_points(rpoints);
        r = rpoints[m];

        //alpha_BA_chi[m] = first_deriv(r, fields->phi);

        K_B_alpha[m]      =  (field_phys_i_sym(m, Bmat, metric->alpha) + 1.0) * field_phys_i_sym(m, Bmat, metric->K_B);
        alpha_K[m]        =  (field_phys_i_sym(m, Bmat, metric->alpha) + 1.0) * field_phys_i_sym(m, Bmat, metric->K);
        alpha_B_over_A[m] =  (field_phys_i_sym(m, Bmat, metric->alpha) + 1.0) * (field_phys_i_sym(m, Bmat, metric->B)+1.0) / (sqrt(field_phys_i_sym(m, Bmat, metric->A)+1.0)) -1.0;
        alpha_BA_chi[m]   =  (field_phys_i_sym(m, Bmat, metric->alpha) + 1.0) * (field_phys_i_sym(m, Bmat, metric->B)+1.0) / (sqrt(field_phys_i_sym(m, Bmat, metric->A)+1.0))*first_deriv_sym(m, C_sym, c_fields->phi);

    }

    dot_product_for_coeff(B_inv_sym,   alpha_B_over_A);
    dot_product_for_coeff(B_inv_asym, alpha_BA_chi);
    dot_product_for_coeff(B_inv_sym,   K_B_alpha);
    dot_product_for_coeff(B_inv_sym,   alpha_K);

        
    #pragma omp parallel for 
    for (int i = 0; i < lattice_size; i++) {
        double r, rpoints[lattice_size];
        make_points(rpoints);
        r = rpoints[i];
        // defining some variables in the physical domain
        double phi, pi;
        double A, B, alpha, D_B, U_tilda, K, K_B, D_alpha, lambda, B_deriv, A_deriv, alpha_deriv, A_second_deriv, B_second_deriv, alpha_second_deriv;
        // and also some of their derivatives
        double phi_deriv, D_B_deriv, K_deriv, K_B_deriv, D_alpha_deriv, lambda_deriv, U_tilda_deriv, alpha_BA_chi_deriv, K_B_alpha_deriv;
        // and one second derivative
        double phi_second_deriv;
        // stress tensor
        double rho = 0.0, j_A = 0.0, S_A = 0.0, S_B = 0.0;

        Stress_Tensor stress_tnsr;
        Bi_Linears    bi_linears;

        set_bi_linears   (i, &bi_linears,  c_fields,   q_fields, metric, Bmat, C_sym);
        set_stress_tensor(i, &stress_tnsr, bi_linears, metric,   c_fields, q_fields, Bmat);

        rho = stress_tnsr.rho;
        j_A = stress_tnsr.j_A;
        S_A = stress_tnsr.S_A;
        S_B = stress_tnsr.S_B;

        // set PHYSICAL DOMAIN variables that we need
        pi                 = field_phys_i_sym  (i,  Bmat,   c_fields->pi);
        phi                = field_phys_i_sym  (i,  Bmat,   c_fields->phi);
        phi_deriv          = first_deriv_sym   (i,  C_sym,  c_fields->phi);
        phi_second_deriv   = second_deriv      (i,  D,      c_fields->phi);
        
        // metric variables
        A           = field_phys_i_sym  (i, Bmat,   metric->A)+1.0;
        B           = field_phys_i_sym  (i, Bmat,   metric->B)+1.0;
        K           = field_phys_i_sym  (i, Bmat,   metric->K);
        K_B         = field_phys_i_sym  (i, Bmat,   metric->K_B);
        alpha       = field_phys_i_sym  (i, Bmat,   metric->alpha)+1.0;
        D_B         = field_phys_i_asym (i, Bmatas, metric->D_B);
        D_alpha     = field_phys_i_asym (i, Bmatas, metric->D_alpha);
  
        lambda      = field_phys_i_asym (i, Bmatas, metric->lambda);
        U_tilda     = field_phys_i_asym (i, Bmatas, metric->U_tilda);

        // and their derivatives that we need
        A_deriv             = first_deriv_sym    (i, C_sym, metric->A);
        B_deriv             = first_deriv_sym    (i, C_sym, metric->B);
        alpha_deriv         = first_deriv_sym    (i, C_sym, metric->alpha);
        K_deriv             = first_deriv_sym    (i, C_sym, metric->K);
        K_B_deriv           = first_deriv_sym    (i, C_sym, metric->K_B);
        D_B_deriv           = first_deriv_asym   (i, C_asym, metric->D_B);
        D_alpha_deriv       = first_deriv_asym   (i, C_asym, metric->D_alpha);
        lambda_deriv        = first_deriv_asym   (i, C_asym, metric->lambda);
        U_tilda_deriv       = first_deriv_asym   (i, C_asym, metric->U_tilda);

 
        alpha_BA_chi_deriv  = first_deriv_asym   (i, C_asym, alpha_BA_chi);
        K_B_alpha_deriv     = first_deriv_sym    (i, C_sym, K_B_alpha);

        rho = 1.0 / (2.0 * A) * (pi * pi / (B * B) + phi_deriv * phi_deriv);
        j_A = -phi_deriv * pi / (sqrt(A) * B);
        S_A = 1.0 / (2.0 * A) * (pi * pi / (B * B) + phi_deriv * phi_deriv);
        S_B = 1.0 / (2.0 * A) * (pi * pi / (B * B) - phi_deriv * phi_deriv);

        // MATTER //

        // classical
        c_fields_dot->phi[i] = alpha / (sqrt(A) * B) * pi;
        c_fields_dot->pi[i] = (i != 0 ? alpha_BA_chi_deriv + 2.0 / r * alpha*B/sqrt(A) * phi_deriv : 3.0 * alpha * B / sqrt(A) * phi_second_deriv);
        // quantum
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                int l_value;
                l_value = l_start + l * l_step;
                for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                    __complex__ double phi_mode = 0.0, chi_mode = 0.0, pi_mode = 0.0;
                    __complex__ double phi_mode_second_deriv = 0.0;

                    phi_mode = field_phys_i_sym_comp(i, Bmat, q_fields[which_q_field]->phi[k][l]);
                    chi_mode = first_deriv_sym_comp(i, C_sym, q_fields[which_q_field]->phi[k][l]);
                    pi_mode = field_phys_i_sym_comp(i, Bmat, q_fields[which_q_field]->pi[k][l]);
                    phi_mode_second_deriv = second_deriv_comp(i, D, q_fields[which_q_field]->phi[k][l]);

                    //q_fields_dot[which_q_field]->phi[k][l][i] = alpha / (sqrt(A) * B) * pi_mode;

                    //q_fields_dot[which_q_field]->pi[k][l][i] = first_deriv_sym(i, C_sym, alpha_B_over_A) * (i != 0 ? (l_value / r * phi_mode + chi_mode) : 0.0)

//                        + alpha * B / sqrt(A) * (i != 0 ? (phi_mode_second_deriv + (2.0 * l_value + 2.0) / r * chi_mode) : ((2.0 * l_value + 3.0) * phi_mode_second_deriv))
                    
                    //    + (i != 0 ? l_value * (l_value + 1) / r * alpha * B / sqrt(A) * lambda * phi_mode : l_value * (l_value + 1) * alpha * B / sqrt(A) * phi_mode * lambda_deriv)

                      //  - alpha * B * sqrt(A) * massSq[which_q_field] * phi_mode;
                    q_fields_dot[which_q_field]->phi[k][l][i] = pi_mode;
                    //q_fields_dot[which_q_field]->chi[k][l][i] = first_deriv_comp(i, q_fields[which_q_field]->pi[k][l]);
                    q_fields_dot[which_q_field]->pi[k][l][i] = (i != 0 ? (phi_mode_second_deriv + (2.0 * l_value + 2.0) / r * chi_mode) : ((2.0 * l_value + 3.0) * phi_mode_second_deriv)) - massSq[which_q_field] * phi_mode;
                }
            }
        }

        metric_dot->A[i] = (-2.0 * alpha * A * (K - 2.0 * K_B));

        metric_dot->B[i] = (-2.0 * alpha * B * K_B);

        metric_dot->D_B[i] = -2.0 * K_B_alpha_deriv;

        metric_dot->U_tilda[i] = -2.0 * alpha * (K_deriv + D_alpha * (K - 4.0 * K_B)
            - 2.0 * (K - 3.0 * K_B) * (D_B - 2.0 * lambda * B / A))
            - 4.0 * alpha * j_A * (M_P * M_P);


        
        metric_dot->K_B[i] = (r != 0.0 ? alpha / (r * A) * (0.5 * U_tilda + 2.0 * lambda * B / A - D_B - lambda - D_alpha) : alpha / (A) * (0.5 * U_tilda_deriv
            + (2.0 * B / A - 1.0) * lambda_deriv
            - D_B_deriv
            - D_alpha_deriv))
            + alpha / A * (-0.5 * D_alpha * D_B
                - 0.5 * D_B_deriv + 0.25 * D_B * (U_tilda + 4.0 * lambda * B / A)
                + A * K * K_B)
            + alpha / (2.0) * M_P * M_P * (S_A - rho + 2.0 * cosm_const);
            
              
                
        metric_dot->K[i] = alpha * (K * K - 4.0 * K * K_B + 6.0 * K_B * K_B)
            - (r != 0.0 ? alpha / A * (D_alpha_deriv + D_alpha * D_alpha + 2.0 * D_alpha / r - 0.5 * D_alpha * (U_tilda + 4.0 * lambda * B / A)) : alpha / A * (3.0 * D_alpha_deriv))
            + alpha / 2.0 * M_P * M_P * (rho + S_A + 2.0 * S_B + 2.0 * cosm_const);
            
            
        metric_dot->lambda[i] = 2.0 * alpha * A / B * (K_B_deriv - 0.5 * D_B * (K - 3.0 * K_B) + 1.0 / 2.0 * M_P * M_P * j_A);
        
        

        metric_dot->alpha[i]   = -2.0 * alpha * K;                               ///// 1+LOG GAUGE
        metric_dot->D_alpha[i] = -2.0 * K_deriv;                               ///// 1+LOG GAUGE
        //metric_dot->alpha[i]   = -alpha*alpha*K;                             ///// HARMONIC GAUGE
        //metric_dot->D_alpha[i] = -first_deriv_sym(i, C_sym, alpha_K);                   ///// HARMONIC GAUGE

    }

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Find basis function coefficients for new time slice functions */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_coeff_dot(double** B, double** B_as, double** B_inv_sym, double** B_inv_asym, double** C_sym, double** C_asym, double** D, 
                    Classical_Fields* c_fields,  Classical_Fields* c_fields_dot, 
                    Quantum_Fields**  q_fields,  Quantum_Fields**  q_fields_dot, 
                    Metric_Fields*    metric,    Metric_Fields*    metric_dot, double cosm_const) {


    // find the f-s
    find_f(c_fields, c_fields_dot, q_fields, q_fields_dot, metric, metric_dot, B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, cosm_const);
    
    // find coeff_dot
    dot_product_for_coeff(B_inv_sym,    c_fields_dot->phi); // d/dt \hat{phi} = f_phi = pi
    dot_product_for_coeff(B_inv_sym,    c_fields_dot->pi);  // d/dt \hat{pi}  = f_pi  = 2/r phi' + phi''

    dot_product_for_coeff(B_inv_sym,    metric_dot->A);
    dot_product_for_coeff(B_inv_sym,    metric_dot->B);
    dot_product_for_coeff(B_inv_asym,   metric_dot->D_B);
    dot_product_for_coeff(B_inv_sym,    metric_dot->K);
    dot_product_for_coeff(B_inv_sym,    metric_dot->K_B);
    dot_product_for_coeff(B_inv_sym,    metric_dot->alpha);
    dot_product_for_coeff(B_inv_asym,   metric_dot->D_alpha);
    dot_product_for_coeff(B_inv_asym,   metric_dot->U_tilda);
    dot_product_for_coeff(B_inv_asym,   metric_dot->lambda);

    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            int l_value;
            l_value = l_start + l * l_step;
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                dot_product_for_coeff_comp(B_inv_sym, q_fields_dot[which_q_field]->phi[k][l]);
                dot_product_for_coeff_comp(B_inv_sym, q_fields_dot[which_q_field]->pi[k][l]);

            }
        }
    }
    
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* One Runge-Kutta time iteration */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void single_RK_convergence_step_RK5(double** B, double** B_as, double** B_inv_sym, double** B_inv_asym, double** C_sym, double** C_asym, double** D, double a_ij[nu_legendre][nu_legendre],
                                    Classical_Fields* c_fields_RK1,     Classical_Fields *c_fields_RK2,     Classical_Fields *c_fields_RK3,     Classical_Fields *c_fields_RK4,     Classical_Fields *c_fields_RK5,     Classical_Fields *c_fields_RK_sum,     Classical_Fields *c_fields,
                                    Quantum_Fields** q_fields_RK1,      Quantum_Fields** q_fields_RK2,      Quantum_Fields** q_fields_RK3,      Quantum_Fields** q_fields_RK4,      Quantum_Fields** q_fields_RK5,      Quantum_Fields** q_fields_RK_sum,      Quantum_Fields** q_fields,
                                    Metric_Fields* metric_RK1, Metric_Fields* metric_RK2, Metric_Fields* metric_RK3, Metric_Fields* metric_RK4, Metric_Fields* metric_RK5, Metric_Fields* metric_RK_sum, Metric_Fields* metric,
                                    double cosm_const)
{

    //first iterate the RK1 term
    #pragma omp parallel for 
    for(int i=0; i<lattice_size; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[0][0]*c_fields_RK1->phi[i] + a_ij[0][1]*c_fields_RK2->phi[i]+a_ij[0][2]*c_fields_RK3->phi[i]+a_ij[0][3]*c_fields_RK4->phi[i]+a_ij[0][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[0][0]*c_fields_RK1->pi[i]  + a_ij[0][1]*c_fields_RK2->pi[i] +a_ij[0][2]*c_fields_RK3->pi[i] +a_ij[0][3]*c_fields_RK4->pi[i] +a_ij[0][4]*c_fields_RK5->pi[i]);

        //#pragma omp parallel for
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[0][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[0][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[0][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[0][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[0][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[0][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[0][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[0][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[0][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[0][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[0][0]*metric_RK1->A[i]      + a_ij[0][1]*metric_RK2->A[i]        + a_ij[0][2]*metric_RK3->A[i]       + a_ij[0][3]*metric_RK4->A[i]       + a_ij[0][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[0][0]*metric_RK1->B[i]      + a_ij[0][1]*metric_RK2->B[i]        + a_ij[0][2]*metric_RK3->B[i]       + a_ij[0][3]*metric_RK4->B[i]       + a_ij[0][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[0][0]*metric_RK1->D_B[i]    + a_ij[0][1]*metric_RK2->D_B[i]      + a_ij[0][2]*metric_RK3->D_B[i]     + a_ij[0][3]*metric_RK4->D_B[i]     + a_ij[0][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[0][0]*metric_RK1->U_tilda[i]+ a_ij[0][1]*metric_RK2->U_tilda[i]  + a_ij[0][2]*metric_RK3->U_tilda[i] + a_ij[0][3]*metric_RK4->U_tilda[i] + a_ij[0][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[0][0]*metric_RK1->K[i]      + a_ij[0][1]*metric_RK2->K[i]        + a_ij[0][2]*metric_RK3->K[i]       + a_ij[0][3]*metric_RK4->K[i]       + a_ij[0][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[0][0]*metric_RK1->K_B[i]    + a_ij[0][1]*metric_RK2->K_B[i]      + a_ij[0][2]*metric_RK3->K_B[i]     + a_ij[0][3]*metric_RK4->K_B[i]     + a_ij[0][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[0][0]*metric_RK1->lambda[i] + a_ij[0][1]*metric_RK2->lambda[i]   + a_ij[0][2]*metric_RK3->lambda[i]  + a_ij[0][3]*metric_RK4->lambda[i]  + a_ij[0][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[0][0]*metric_RK1->alpha[i]  + a_ij[0][1]*metric_RK2->alpha[i]    + a_ij[0][2]*metric_RK3->alpha[i]   + a_ij[0][3]*metric_RK4->alpha[i]   + a_ij[0][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[0][0]*metric_RK1->D_alpha[i]+ a_ij[0][1]*metric_RK2->D_alpha[i]  + a_ij[0][2]*metric_RK3->D_alpha[i] + a_ij[0][3]*metric_RK4->D_alpha[i] + a_ij[0][4]*metric_RK5->D_alpha[i]);

    }
    
    find_coeff_dot(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, c_fields_RK_sum, c_fields_RK1, q_fields_RK_sum, q_fields_RK1, metric_RK_sum, metric_RK1, cosm_const);

    //then iterate the RK2 term
    #pragma omp parallel for 
    for(int i=0; i<lattice_size; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[1][0]*c_fields_RK1->phi[i] + a_ij[1][1]*c_fields_RK2->phi[i]+a_ij[1][2]*c_fields_RK3->phi[i]+a_ij[1][3]*c_fields_RK4->phi[i]+a_ij[1][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[1][0]*c_fields_RK1->pi[i]  + a_ij[1][1]*c_fields_RK2->pi[i] +a_ij[1][2]*c_fields_RK3->pi[i] +a_ij[1][3]*c_fields_RK4->pi[i] +a_ij[1][4]*c_fields_RK5->pi[i]);

        // #pragma omp parallel for
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[1][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[1][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[1][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[1][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[1][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[1][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[1][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[1][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[1][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[1][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[1][0]*metric_RK1->A[i]      + a_ij[1][1]*metric_RK2->A[i]        + a_ij[1][2]*metric_RK3->A[i]       + a_ij[1][3]*metric_RK4->A[i]       + a_ij[1][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[1][0]*metric_RK1->B[i]      + a_ij[1][1]*metric_RK2->B[i]        + a_ij[1][2]*metric_RK3->B[i]       + a_ij[1][3]*metric_RK4->B[i]       + a_ij[1][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[1][0]*metric_RK1->D_B[i]    + a_ij[1][1]*metric_RK2->D_B[i]      + a_ij[1][2]*metric_RK3->D_B[i]     + a_ij[1][3]*metric_RK4->D_B[i]     + a_ij[1][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[1][0]*metric_RK1->U_tilda[i]+ a_ij[1][1]*metric_RK2->U_tilda[i]  + a_ij[1][2]*metric_RK3->U_tilda[i] + a_ij[1][3]*metric_RK4->U_tilda[i] + a_ij[1][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[1][0]*metric_RK1->K[i]      + a_ij[1][1]*metric_RK2->K[i]        + a_ij[1][2]*metric_RK3->K[i]       + a_ij[1][3]*metric_RK4->K[i]       + a_ij[1][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[1][0]*metric_RK1->K_B[i]    + a_ij[1][1]*metric_RK2->K_B[i]      + a_ij[1][2]*metric_RK3->K_B[i]     + a_ij[1][3]*metric_RK4->K_B[i]     + a_ij[1][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[1][0]*metric_RK1->lambda[i] + a_ij[1][1]*metric_RK2->lambda[i]   + a_ij[1][2]*metric_RK3->lambda[i]  + a_ij[1][3]*metric_RK4->lambda[i]  + a_ij[1][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[1][0]*metric_RK1->alpha[i]  + a_ij[1][1]*metric_RK2->alpha[i]    + a_ij[1][2]*metric_RK3->alpha[i]   + a_ij[1][3]*metric_RK4->alpha[i]   + a_ij[1][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[1][0]*metric_RK1->D_alpha[i]+ a_ij[1][1]*metric_RK2->D_alpha[i]  + a_ij[1][2]*metric_RK3->D_alpha[i] + a_ij[1][3]*metric_RK4->D_alpha[i] + a_ij[1][4]*metric_RK5->D_alpha[i]);

    }
    find_coeff_dot(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, c_fields_RK_sum, c_fields_RK2, q_fields_RK_sum, q_fields_RK2, metric_RK_sum, metric_RK2, cosm_const);
    //then iterate the RK3 term
    #pragma omp parallel for 
    for(int i=0; i<lattice_size; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[2][0]*c_fields_RK1->phi[i] + a_ij[2][1]*c_fields_RK2->phi[i]+a_ij[2][2]*c_fields_RK3->phi[i]+a_ij[2][3]*c_fields_RK4->phi[i]+a_ij[2][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[2][0]*c_fields_RK1->pi[i]  + a_ij[2][1]*c_fields_RK2->pi[i] +a_ij[2][2]*c_fields_RK3->pi[i] +a_ij[2][3]*c_fields_RK4->pi[i] +a_ij[2][4]*c_fields_RK5->pi[i]);

        //#pragma omp parallel for
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[2][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[2][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[2][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[2][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[2][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[2][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[2][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[2][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[2][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[2][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[2][0]*metric_RK1->A[i]      + a_ij[2][1]*metric_RK2->A[i]        + a_ij[2][2]*metric_RK3->A[i]       + a_ij[2][3]*metric_RK4->A[i]       + a_ij[2][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[2][0]*metric_RK1->B[i]      + a_ij[2][1]*metric_RK2->B[i]        + a_ij[2][2]*metric_RK3->B[i]       + a_ij[2][3]*metric_RK4->B[i]       + a_ij[2][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[2][0]*metric_RK1->D_B[i]    + a_ij[2][1]*metric_RK2->D_B[i]      + a_ij[2][2]*metric_RK3->D_B[i]     + a_ij[2][3]*metric_RK4->D_B[i]     + a_ij[2][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[2][0]*metric_RK1->U_tilda[i]+ a_ij[2][1]*metric_RK2->U_tilda[i]  + a_ij[2][2]*metric_RK3->U_tilda[i] + a_ij[2][3]*metric_RK4->U_tilda[i] + a_ij[2][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[2][0]*metric_RK1->K[i]      + a_ij[2][1]*metric_RK2->K[i]        + a_ij[2][2]*metric_RK3->K[i]       + a_ij[2][3]*metric_RK4->K[i]       + a_ij[2][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[2][0]*metric_RK1->K_B[i]    + a_ij[2][1]*metric_RK2->K_B[i]      + a_ij[2][2]*metric_RK3->K_B[i]     + a_ij[2][3]*metric_RK4->K_B[i]     + a_ij[2][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[2][0]*metric_RK1->lambda[i] + a_ij[2][1]*metric_RK2->lambda[i]   + a_ij[2][2]*metric_RK3->lambda[i]  + a_ij[2][3]*metric_RK4->lambda[i]  + a_ij[2][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[2][0]*metric_RK1->alpha[i]  + a_ij[2][1]*metric_RK2->alpha[i]    + a_ij[2][2]*metric_RK3->alpha[i]   + a_ij[2][3]*metric_RK4->alpha[i]   + a_ij[2][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[2][0]*metric_RK1->D_alpha[i]+ a_ij[2][1]*metric_RK2->D_alpha[i]  + a_ij[2][2]*metric_RK3->D_alpha[i] + a_ij[2][3]*metric_RK4->D_alpha[i] + a_ij[2][4]*metric_RK5->D_alpha[i]);

    }
    find_coeff_dot(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, c_fields_RK_sum, c_fields_RK3, q_fields_RK_sum, q_fields_RK3, metric_RK_sum, metric_RK3, cosm_const);
    //then iterate the RK4 term 
    #pragma omp parallel for 
    for(int i=0; i<lattice_size; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[3][0]*c_fields_RK1->phi[i] + a_ij[3][1]*c_fields_RK2->phi[i]+a_ij[3][2]*c_fields_RK3->phi[i]+a_ij[3][3]*c_fields_RK4->phi[i]+a_ij[3][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[3][0]*c_fields_RK1->pi[i]  + a_ij[3][1]*c_fields_RK2->pi[i] +a_ij[3][2]*c_fields_RK3->pi[i] +a_ij[3][3]*c_fields_RK4->pi[i] +a_ij[3][4]*c_fields_RK5->pi[i]);

        //#pragma omp parallel for
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[3][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[3][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[3][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[3][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[3][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[3][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[3][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[3][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[3][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[3][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[3][0]*metric_RK1->A[i]      + a_ij[3][1]*metric_RK2->A[i]        + a_ij[3][2]*metric_RK3->A[i]       + a_ij[3][3]*metric_RK4->A[i]       + a_ij[3][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[3][0]*metric_RK1->B[i]      + a_ij[3][1]*metric_RK2->B[i]        + a_ij[3][2]*metric_RK3->B[i]       + a_ij[3][3]*metric_RK4->B[i]       + a_ij[3][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[3][0]*metric_RK1->D_B[i]    + a_ij[3][1]*metric_RK2->D_B[i]      + a_ij[3][2]*metric_RK3->D_B[i]     + a_ij[3][3]*metric_RK4->D_B[i]     + a_ij[3][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[3][0]*metric_RK1->U_tilda[i]+ a_ij[3][1]*metric_RK2->U_tilda[i]  + a_ij[3][2]*metric_RK3->U_tilda[i] + a_ij[3][3]*metric_RK4->U_tilda[i] + a_ij[3][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[3][0]*metric_RK1->K[i]      + a_ij[3][1]*metric_RK2->K[i]        + a_ij[3][2]*metric_RK3->K[i]       + a_ij[3][3]*metric_RK4->K[i]       + a_ij[3][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[3][0]*metric_RK1->K_B[i]    + a_ij[3][1]*metric_RK2->K_B[i]      + a_ij[3][2]*metric_RK3->K_B[i]     + a_ij[3][3]*metric_RK4->K_B[i]     + a_ij[3][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[3][0]*metric_RK1->lambda[i] + a_ij[3][1]*metric_RK2->lambda[i]   + a_ij[3][2]*metric_RK3->lambda[i]  + a_ij[3][3]*metric_RK4->lambda[i]  + a_ij[3][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[3][0]*metric_RK1->alpha[i]  + a_ij[3][1]*metric_RK2->alpha[i]    + a_ij[3][2]*metric_RK3->alpha[i]   + a_ij[3][3]*metric_RK4->alpha[i]   + a_ij[3][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[3][0]*metric_RK1->D_alpha[i]+ a_ij[3][1]*metric_RK2->D_alpha[i]  + a_ij[3][2]*metric_RK3->D_alpha[i] + a_ij[3][3]*metric_RK4->D_alpha[i] + a_ij[3][4]*metric_RK5->D_alpha[i]);

    }
    find_coeff_dot(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, c_fields_RK_sum, c_fields_RK4, q_fields_RK_sum, q_fields_RK4, metric_RK_sum, metric_RK4, cosm_const);
    //then iterate the RK5 term
    #pragma omp parallel for 
    for(int i=0; i<lattice_size; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[4][0]*c_fields_RK1->phi[i] + a_ij[4][1]*c_fields_RK2->phi[i]+a_ij[4][2]*c_fields_RK3->phi[i]+a_ij[4][3]*c_fields_RK4->phi[i]+a_ij[4][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[4][0]*c_fields_RK1->pi[i]  + a_ij[4][1]*c_fields_RK2->pi[i] +a_ij[4][2]*c_fields_RK3->pi[i] +a_ij[4][3]*c_fields_RK4->pi[i] +a_ij[4][4]*c_fields_RK5->pi[i]);

        //#pragma omp parallel for
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[4][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[4][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[4][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[4][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[4][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[4][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[4][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[4][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[4][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[4][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[4][0]*metric_RK1->A[i]      + a_ij[4][1]*metric_RK2->A[i]        + a_ij[4][2]*metric_RK3->A[i]       + a_ij[4][3]*metric_RK4->A[i]       + a_ij[4][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[4][0]*metric_RK1->B[i]      + a_ij[4][1]*metric_RK2->B[i]        + a_ij[4][2]*metric_RK3->B[i]       + a_ij[4][3]*metric_RK4->B[i]       + a_ij[4][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[4][0]*metric_RK1->D_B[i]    + a_ij[4][1]*metric_RK2->D_B[i]      + a_ij[4][2]*metric_RK3->D_B[i]     + a_ij[4][3]*metric_RK4->D_B[i]     + a_ij[4][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[4][0]*metric_RK1->U_tilda[i]+ a_ij[4][1]*metric_RK2->U_tilda[i]  + a_ij[4][2]*metric_RK3->U_tilda[i] + a_ij[4][3]*metric_RK4->U_tilda[i] + a_ij[4][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[4][0]*metric_RK1->K[i]      + a_ij[4][1]*metric_RK2->K[i]        + a_ij[4][2]*metric_RK3->K[i]       + a_ij[4][3]*metric_RK4->K[i]       + a_ij[4][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[4][0]*metric_RK1->K_B[i]    + a_ij[4][1]*metric_RK2->K_B[i]      + a_ij[4][2]*metric_RK3->K_B[i]     + a_ij[4][3]*metric_RK4->K_B[i]     + a_ij[4][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[4][0]*metric_RK1->lambda[i] + a_ij[4][1]*metric_RK2->lambda[i]   + a_ij[4][2]*metric_RK3->lambda[i]  + a_ij[4][3]*metric_RK4->lambda[i]  + a_ij[4][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[4][0]*metric_RK1->alpha[i]  + a_ij[4][1]*metric_RK2->alpha[i]    + a_ij[4][2]*metric_RK3->alpha[i]   + a_ij[4][3]*metric_RK4->alpha[i]   + a_ij[4][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[4][0]*metric_RK1->D_alpha[i]+ a_ij[4][1]*metric_RK2->D_alpha[i]  + a_ij[4][2]*metric_RK3->D_alpha[i] + a_ij[4][3]*metric_RK4->D_alpha[i] + a_ij[4][4]*metric_RK5->D_alpha[i]);

    }
    find_coeff_dot(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, c_fields_RK_sum, c_fields_RK5, q_fields_RK_sum, q_fields_RK5, metric_RK_sum, metric_RK5, cosm_const);
    
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves arrays of lattice size*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_points(){
    double r[lattice_size];
    make_points(r);
    FILE * pointsout;
    pointsout=fopen("points10.txt", "w");
    for (int i=0;i<lattice_size;++i){
        fprintf(pointsout,"%.40f ",r[i]);
    }
    fclose(pointsout);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves matrices*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_field(double *field){

    FILE * finout;
    finout=fopen("data10.txt", "w");
    for (int m=0;m<lattice_size;++m){
        fprintf(finout, "%.200f ", field[m]);
    }
    fclose(finout);
}
void save_field_comp(__complex__ double* field) {

    FILE* finout;
    finout = fopen("data10c.txt", "w");
    for (int m = 0; m < lattice_size; ++m) {
        fprintf(finout, "%.200f ", __real__ field[m]);
    }
    fclose(finout);
}
void save_field1( double *field){

    FILE * finout;
    finout=fopen("ham_r_gal.txt", "w");
    for (int m=0;m<lattice_size;++m){
        fprintf(finout, "%.100f ", field[m]);
    }
    fclose(finout);
}
void save_field2(double *field){

    FILE * finout;
    finout=fopen("data12.txt", "w");
    for (int m=0;m<lattice_size;++m){
        fprintf(finout, "%.100f ", field[m]);
    }
    fclose(finout);
}
void save_field3(double *field){

    FILE * finout;
    finout=fopen("data13.txt", "w");
    for (int m=0;m<lattice_size;++m){
        fprintf(finout, "%.100f ", field[m]);
    }
    fclose(finout);
}
void save_alpha_t(double* field) {

    FILE* finout;
    finout = fopen("alpha_t.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_hor_t(double* field) {

    FILE* finout;
    finout = fopen("horizon.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_ham_t(double* field) {

    FILE* finout;
    finout = fopen("ham_t_gal150bh.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves all the stress tensor components */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_stress_tensor(int n, double cos_const, Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric, double** Bmat, double** C) {
    double r[lattice_size];
    make_points(r);
    double rho[lattice_size], j_A[lattice_size], S_A[lattice_size], S_B[lattice_size];

    //#pragma omp parallel for //num_threads(lattice_size)
    for (int i = 0; i < lattice_size; i++) {

        double A, B;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        Bi_Linears    bi_linears;


        A = field_phys_i_sym(i, Bmat, metric->A)+1.0;
        B = field_phys_i_sym(i, Bmat, metric->B)+1.0;

        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric, Bmat, C);

        phi_phi = bi_linears.phi_phi;
        chi_chi = bi_linears.chi_chi;
        pi_pi = bi_linears.pi_pi;
        chi_pi =  bi_linears.chi_pi;
        del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

        rho[i] =  1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq - cos_const;
        j_A[i] =  -chi_pi / (sqrt(A) * B);
        S_A[i] =  1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq + cos_const;
        S_B[i] =  1.0 / (2.0 * A) * (pi_pi / (B * B) - chi_chi) + cos_const;


    }
    // saving stress-energy tensor for different time steps
    if (n == 0) {
        FILE* finout;
        finout = fopen("T0c_00.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.100f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("T0c_01.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout1, "%.100f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("T0c_11.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout2, "%.100f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("T0c_22.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout3, "%.100f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 1) {
        FILE* finout;
        finout = fopen("T1c_00.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.100f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("T1c_01.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout1, "%.100f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("T1c_11.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout2, "%.100f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("T1c_22.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout3, "%.100f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 2) {
        FILE* finout;
        finout = fopen("T2_00.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.100f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("T2_01.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout1, "%.100f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("T2_11.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout2, "%.100f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("T2_22.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout3, "%.100f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 3) {
        FILE* finout;
        finout = fopen("T3_00.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.100f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("T3_01.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout1, "%.100f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("T3_11.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout2, "%.100f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("T3_22.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout3, "%.100f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 4) {
        FILE* finout;
        finout = fopen("T4_00.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.100f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("T4_01.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout1, "%.100f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("T4_11.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout2, "%.100f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("T4_22.txt", "w");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout3, "%.100f ", S_B[m]);
        }
        fclose(finout3);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the L_2 norm of the Hamiltonian constraint for each time step */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void hamiltonian_constraint(int n, double** Bmat, double** Bmatas, double** C_sym, double** C_asym, double** D, Classical_Fields* c_fields, Metric_Fields* metric, double ham[evolve_time_int]) {

    double rpoints[lattice_size];
    make_points(rpoints);

    //#pragma omp parallel for
//#pragma omp parallel for 
    for (int i = 0; i < lattice_size; i++) {
        double phi, pi, A, B, K, K_B, alpha, D_B, D_alpha, lambda, U_tilda;
        double chi, A_deriv, B_deriv, alpha_deriv, D_B_deriv, D_alpha_deriv, lambda_deriv, U_tilda_deriv;
        double A_second_deriv, B_second_deriv, alpha_second_deriv;
        double rho, r;

        r = rpoints[i];

        pi                 = field_phys_i_sym  (i,  Bmat,   c_fields->pi);
        phi                = field_phys_i_sym  (i,  Bmat,   c_fields->phi);
        chi                = first_deriv_sym   (i, C_sym,   c_fields->phi);

        A           = field_phys_i_sym  (i, Bmat,   metric->A)+1.0;
        B           = field_phys_i_sym  (i, Bmat,   metric->B)+1.0;
        K           = field_phys_i_sym  (i, Bmat,   metric->K);
        K_B         = field_phys_i_sym  (i, Bmat,   metric->K_B);
        alpha       = field_phys_i_sym  (i, Bmat,   metric->alpha)+1.0;

        A_deriv             = first_deriv_sym    (i, C_sym, metric->A);
        B_deriv             = first_deriv_sym    (i, C_sym, metric->B);
        alpha_deriv         = first_deriv_sym    (i, C_sym, metric->alpha);
        A_second_deriv      = second_deriv       (i, D, metric->A);
        B_second_deriv      = second_deriv       (i, D, metric->B);
        alpha_second_deriv  = second_deriv       (i, D, metric->alpha);

        //D_B = B_deriv / B;
        //D_alpha = alpha_deriv / alpha;
        D_B         = field_phys_i_asym (i, Bmatas, metric->D_B);
        D_alpha     = field_phys_i_asym (i, Bmatas, metric->D_alpha);
        lambda      = field_phys_i_asym (i, Bmatas, metric->lambda);
        U_tilda     = field_phys_i_asym (i, Bmatas, metric->U_tilda);


        //D_B_deriv          = (i != 0 ? B_second_deriv / (B)       - D_B*D_B         : B_second_deriv / B);
        //D_alpha_deriv      = (i != 0 ? alpha_second_deriv/(alpha) - D_alpha*D_alpha : alpha_second_deriv / alpha);
        //lambda_deriv       = (i != 0 ? -A_deriv/(r*B)+A*B_deriv/(r*B*B)+A/(r*r*B)-1.0/(r*r) : A_second_deriv/B+A*B_second_deriv/(B*B));
        //U_tilda_deriv      =  A_second_deriv/(A) - A_deriv*A_deriv/(A*A) - 2.0*D_B_deriv - 4.0*(B_deriv*lambda/A + lambda_deriv*B/A - B*lambda*A_deriv/(A*A));

        D_B_deriv = first_deriv_asym(i, C_asym, metric->D_B);
        D_alpha_deriv = first_deriv_asym(i, C_asym, metric->D_alpha);
        lambda_deriv = first_deriv_asym(i, C_asym, metric->lambda);
        U_tilda_deriv = first_deriv_asym(i, C_asym, metric->U_tilda);

        rho = 1.0 / (2.0 * A) * (pi*pi / (B * B) + chi*chi);

        ham[n] = ham[n] + pow((D_B_deriv
            + (r != 0.0 ? 1 / r * (lambda + D_B - U_tilda - 4.0 * lambda*B/A)
                : lambda_deriv + D_B_deriv - U_tilda_deriv - 4.0 * (B_deriv*lambda/A + lambda_deriv*B/A - B*lambda*A_deriv/(A*A)))
            - D_B * (0.25 * D_B + 0.5 * U_tilda + 2.0 * lambda*B/A)
            - A * K_B * (2.0*K - 3.0*K_B)
            + A * (rho) * M_P * M_P), 2);

    }
    ham[n] = sqrt((ham[n])) / lattice_size;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the Hamiltonian constraint at one time step for all r*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ham_const_r(double** Bmat, double** Bmatas, double** C_sym, double** C_asym, double** D, Classical_Fields* c_fields, Metric_Fields* metric, double ham[lattice_size]) {
    
    double rpoints[lattice_size];
    make_points(rpoints);
    for (int i = 0; i < lattice_size; i++) {
        double phi, pi, A, B, K, K_B, alpha, D_B, D_alpha, lambda, U_tilda;
        double chi, A_deriv, B_deriv, alpha_deriv, D_B_deriv, D_alpha_deriv, lambda_deriv, U_tilda_deriv;
        double A_second_deriv, B_second_deriv, alpha_second_deriv;
        double rho, r;

        r = rpoints[i];

        pi  = field_phys_i_sym(i, Bmat, c_fields->pi);
        phi = field_phys_i_sym(i, Bmat, c_fields->phi);
        chi = first_deriv_sym(i, C_sym, c_fields->phi);

        A     = field_phys_i_sym(i, Bmat, metric->A)+1.0;
        B     = field_phys_i_sym(i, Bmat, metric->B)+1.0;
        K     = field_phys_i_sym(i, Bmat, metric->K);
        K_B   = field_phys_i_sym(i, Bmat, metric->K_B);
        alpha = field_phys_i_sym(i, Bmat, metric->alpha)+1.0;

        A_deriv            = first_deriv_sym(i, C_sym, metric->A);
        B_deriv            = first_deriv_sym(i, C_sym, metric->B);
        alpha_deriv        = first_deriv_sym(i, C_sym, metric->alpha);
        A_second_deriv     = second_deriv(i, D, metric->A);
        B_second_deriv     = second_deriv(i, D, metric->B);
        alpha_second_deriv = second_deriv(i, D, metric->alpha);

        //D_B = B_deriv / B;
        //D_alpha = alpha_deriv / alpha;
        D_B     = field_phys_i_asym(i, Bmatas, metric->D_B);
        D_alpha = field_phys_i_asym(i, Bmatas, metric->D_alpha);
        lambda  = field_phys_i_asym(i, Bmatas, metric->lambda);
        U_tilda = field_phys_i_asym(i, Bmatas, metric->U_tilda);


        D_B_deriv     = first_deriv_asym(i, C_asym, metric->D_B);// (i != 0 ? B_second_deriv / (B)-D_B * D_B : B_second_deriv / B);
        D_alpha_deriv = first_deriv_asym(i, C_asym, metric->D_alpha); //(i != 0 ? alpha_second_deriv / (alpha)-D_alpha * D_alpha : alpha_second_deriv / alpha);
        lambda_deriv  = first_deriv_asym(i, C_asym, metric->lambda);
        U_tilda_deriv = first_deriv_asym(i, C_asym, metric->U_tilda);
        
        //D_B_deriv     = (i != 0 ? B_second_deriv / (B)-D_B * D_B : B_second_deriv / B);
        //D_alpha_deriv = (i != 0 ? alpha_second_deriv / (alpha)-D_alpha * D_alpha : alpha_second_deriv / alpha);
        //lambda_deriv = (i != 0 ? -A_deriv / (r * B) + A * B_deriv / (r * B * B) + A / (r * r * B) - 1.0 / (r * r) : A_second_deriv / B + A * B_second_deriv / (B * B));
        //U_tilda_deriv = A_second_deriv / (A)-A_deriv * A_deriv / (A * A) - 2.0 * D_B_deriv - 4.0 * (B_deriv * lambda / A + lambda_deriv * B / A - B * lambda * A_deriv / (A * A));

        
        rho = 1.0 / (2.0 * A) * (pi * pi / (B * B) + chi * chi);
        
        ham[i] = D_B_deriv + (r != 0.0 ? 1 / r *( lambda + D_B - U_tilda - 4.0 * lambda * B / A)
            : lambda_deriv + D_B_deriv - U_tilda_deriv - 4.0 * (B_deriv * lambda / A + lambda_deriv * B / A - B * lambda * A_deriv / (A * A)))
            - D_B * (0.25 * D_B + 0.5 * U_tilda + 2.0 * lambda * B / A)
            - A * K_B * (2.0 * K - 3.0 * K_B)
            + A * (rho)*M_P * M_P;
            /*
        ham[i] = (r != 0.0 ? 1 / r * (lambda - U_tilda - 4.0 * lambda / A)
            : lambda_deriv - U_tilda_deriv - 4.0 * (lambda_deriv / A - lambda * A_deriv / (A * A)))
            + A * (rho)*M_P * M_P;*/
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Full evolution to evolve all fields from t=0 to t=t_max */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fullEvolution(double** B, double** B_as, double** B_inv_sym, double** B_inv_asym, double** C_sym, double** C_asym, double** D, double b_i[nu_legendre], double a_ij[nu_legendre][nu_legendre],
                  Classical_Fields *c_fields_RK1,     Classical_Fields *c_fields_RK2,     Classical_Fields *c_fields_RK3,     Classical_Fields *c_fields_RK4,     Classical_Fields *c_fields_RK5,     Classical_Fields *c_fields_RK_sum,     Classical_Fields *c_fields, 
                  Quantum_Fields**   q_fields_RK1,    Quantum_Fields** q_fields_RK2,      Quantum_Fields** q_fields_RK3,      Quantum_Fields** q_fields_RK4,      Quantum_Fields** q_fields_RK5,      Quantum_Fields** q_fields_RK_sum,      Quantum_Fields** q_fields,
                  Metric_Fields *metric_RK1,          Metric_Fields* metric_RK2,          Metric_Fields* metric_RK3,          Metric_Fields* metric_RK4,          Metric_Fields* metric_RK5,          Metric_Fields* metric_RK_sum,          Metric_Fields* metric,
                  double field_save[evolve_time_int/step+1][lattice_size], double alpha_save[evolve_time_int], double apparent_horizon[evolve_time_int], double ham[evolve_time_int], double ham_r[lattice_size], double cosm_const){

    
    int k=1;
    for (int m = 0; m < lattice_size; ++m) {
        field_save[0][m] = c_fields->phi[m];
        metric->A[m] = metric->A[m] - 1.0;
        metric->B[m] = metric->B[m] - 1.0;
        metric->alpha[m] = metric->alpha[m] - 1.0;
    }

 
    // change to phase space
    dot_product_for_coeff(B_inv_sym, c_fields->phi);
    dot_product_for_coeff(B_inv_sym, c_fields->pi);

    dot_product_for_coeff(B_inv_sym,  metric->A);
    dot_product_for_coeff(B_inv_sym,  metric->B);
    dot_product_for_coeff(B_inv_asym, metric->D_B);
    dot_product_for_coeff(B_inv_sym,  metric->K);
    dot_product_for_coeff(B_inv_sym,  metric->K_B);
    dot_product_for_coeff(B_inv_sym,  metric->alpha);
    dot_product_for_coeff(B_inv_asym, metric->D_alpha);
    dot_product_for_coeff(B_inv_asym, metric->U_tilda);
    dot_product_for_coeff(B_inv_asym, metric->lambda);

    #pragma omp parallel for
    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            int l_value;
            l_value = l_start + l * l_step;
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                dot_product_for_coeff_comp(B_inv_sym, q_fields[which_q_field]->phi[k][l]);
                dot_product_for_coeff_comp(B_inv_sym, q_fields[which_q_field]->pi[k][l]);

            }
        }
    }
    //hamiltonian_constraint(0, B, B_as, C_sym, C_asym, D, c_fields, metric, ham);
    save_stress_tensor(0, cosm_const, c_fields, q_fields, metric, B, C_sym);
    
    for (int n = 0; n<evolve_time_int; ++n){
            
            //initialise RK c_fields to zero
            #pragma omp parallel for 
            for (int i = 0; i < lattice_size; ++i) {
                c_fields_RK1->phi[i] = 0.0;
                c_fields_RK2->phi[i] = 0.0;
                c_fields_RK3->phi[i] = 0.0;
                c_fields_RK4->phi[i] = 0.0;
                c_fields_RK5->phi[i] = 0.0;

                c_fields_RK1->pi[i] = 0.0;
                c_fields_RK2->pi[i] = 0.0;
                c_fields_RK3->pi[i] = 0.0;
                c_fields_RK4->pi[i] = 0.0;
                c_fields_RK5->pi[i] = 0.0;

                //#pragma omp parallel for
                for (int l = 0; l < number_of_l_modes; ++l) {
                    //printf("i=%d, k=%d and process: %d\n",i, k, omp_get_thread_num());
                    for (int k = 0; k < number_of_k_modes; ++k) {
                        for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                            q_fields_RK1[which_q_field]->phi[k][l][i] = 0.0;
                            q_fields_RK2[which_q_field]->phi[k][l][i] = 0.0;
                            q_fields_RK3[which_q_field]->phi[k][l][i] = 0.0;
                            q_fields_RK4[which_q_field]->phi[k][l][i] = 0.0;
                            q_fields_RK5[which_q_field]->phi[k][l][i] = 0.0;

                            q_fields_RK1[which_q_field]->pi[k][l][i] = 0.0;
                            q_fields_RK2[which_q_field]->pi[k][l][i] = 0.0;
                            q_fields_RK3[which_q_field]->pi[k][l][i] = 0.0;
                            q_fields_RK4[which_q_field]->pi[k][l][i] = 0.0;
                            q_fields_RK5[which_q_field]->pi[k][l][i] = 0.0;
                        }
                    }
                }

                metric_RK1->A[i] = 0.0;//metric->A[i];
                metric_RK2->A[i] = 0.0;//metric->A[i];
                metric_RK3->A[i] = 0.0;//metric->A[i];
                metric_RK4->A[i] = 0.0;//metric->A[i];
                metric_RK5->A[i] = 0.0;//metric->A[i];

                metric_RK1->B[i] = 0.0;//metric->B[i];
                metric_RK2->B[i] = 0.0;//metric->B[i];
                metric_RK3->B[i] = 0.0;//metric->B[i];
                metric_RK4->B[i] = 0.0;//metric->B[i];
                metric_RK5->B[i] = 0.0;//metric->B[i];

                metric_RK1->D_B[i] = 0.0;//metric->D_B[i];
                metric_RK2->D_B[i] = 0.0;//metric->D_B[i];
                metric_RK3->D_B[i] = 0.0;//metric->D_B[i];
                metric_RK4->D_B[i] = 0.0;//metric->D_B[i];
                metric_RK5->D_B[i] = 0.0;//metric->D_B[i];

                metric_RK1->U_tilda[i] = 0.0;//metric->U_tilda[i];
                metric_RK2->U_tilda[i] = 0.0;//metric->U_tilda[i];
                metric_RK3->U_tilda[i] = 0.0;//metric->U_tilda[i];
                metric_RK4->U_tilda[i] = 0.0;//metric->U_tilda[i];   
                metric_RK5->U_tilda[i] = 0.0;//metric->U_tilda[i];

                metric_RK1->K[i] = 0.0;//metric->K[i];
                metric_RK2->K[i] = 0.0;//metric->K[i];
                metric_RK3->K[i] = 0.0;//metric->K[i];
                metric_RK4->K[i] = 0.0;//metric->K[i];
                metric_RK5->K[i] = 0.0;//metric->K[i];

                metric_RK1->K_B[i] = 0.0;//metric->K_B[i];
                metric_RK2->K_B[i] = 0.0;//metric->K_B[i];
                metric_RK3->K_B[i] = 0.0;//metric->K_B[i];
                metric_RK4->K_B[i] = 0.0;//metric->K_B[i];
                metric_RK5->K_B[i] = 0.0;//metric->K_B[i];

                metric_RK1->lambda[i] = 0.0;//metric->lambda[i];
                metric_RK2->lambda[i] = 0.0;//metric->lambda[i];
                metric_RK3->lambda[i] = 0.0;//metric->lambda[i];
                metric_RK4->lambda[i] = 0.0;//metric->lambda[i];
                metric_RK5->lambda[i] = 0.0;//metric->lambda[i];

                metric_RK1->alpha[i] = 0.0;//metric->alpha[i];
                metric_RK2->alpha[i] = 0.0;//metric->alpha[i];
                metric_RK3->alpha[i] = 0.0;//metric->alpha[i];
                metric_RK4->alpha[i] = 0.0;//metric->alpha[i];
                metric_RK5->alpha[i] = 0.0;//metric->alpha[i];

                metric_RK1->D_alpha[i] = 0.0;//metric->D_alpha[i];
                metric_RK2->D_alpha[i] = 0.0;//metric->D_alpha[i];
                metric_RK3->D_alpha[i] = 0.0;//metric->D_alpha[i];
                metric_RK4->D_alpha[i] = 0.0;//metric->D_alpha[i];
                metric_RK5->D_alpha[i] = 0.0;//metric->D_alpha[i];
            }

            //do a few RK iterations in order to converge on the implicit solution
            for(int iter=0;iter<number_of_RK_implicit_iterations;++iter){
                single_RK_convergence_step_RK5(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, a_ij, 
                                                                               c_fields_RK1, c_fields_RK2, c_fields_RK3, c_fields_RK4, c_fields_RK5, c_fields_RK_sum, c_fields,
                                                                               q_fields_RK1, q_fields_RK2, q_fields_RK3, q_fields_RK4, q_fields_RK5, q_fields_RK_sum, q_fields,
                                                                               metric_RK1,   metric_RK2,   metric_RK3,   metric_RK4,   metric_RK5,   metric_RK_sum,   metric,
                                                                               cosm_const);
            }
            //add up the RK contributions
            #pragma omp parallel for 
            for(int i=0; i<lattice_size; ++i){
                c_fields->phi[i]      = c_fields->phi[i]     + dt*(b_i[0]*c_fields_RK1->phi[i]    + b_i[1]*c_fields_RK2->phi[i]     + b_i[2]*c_fields_RK3->phi[i]      + b_i[3]*c_fields_RK4->phi[i]     + b_i[4]*c_fields_RK5->phi[i]);
                c_fields->pi[i]       = c_fields->pi[i]      + dt*(b_i[0]*c_fields_RK1->pi[i]     + b_i[1]*c_fields_RK2->pi[i]      + b_i[2]*c_fields_RK3->pi[i]       + b_i[3]*c_fields_RK4->pi[i]      + b_i[4]*c_fields_RK5->pi[i]);

                //#pragma omp parallel for
                for(int l=0; l<number_of_l_modes; ++l){
                    for(int k=0; k<number_of_k_modes; ++k){
                        for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                            q_fields[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt*(b_i[0]*q_fields_RK1[which_q_field]->phi[k][l][i]+b_i[1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                        +b_i[2]*q_fields_RK3[which_q_field]->phi[k][l][i]+b_i[3]*q_fields_RK4[which_q_field]->phi[k][l][i]+b_i[4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                            q_fields[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt*(b_i[0]*q_fields_RK1[which_q_field]->pi[k][l][i] +b_i[1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                        +b_i[2]*q_fields_RK3[which_q_field]->pi[k][l][i] +b_i[3]*q_fields_RK4[which_q_field]->pi[k][l][i] +b_i[4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                        }
                    }
                }

                metric->A[i]        = metric->A[i]       + dt*(b_i[0]*metric_RK1->A[i]      + b_i[1]*metric_RK2->A[i]        + b_i[2]*metric_RK3->A[i]       + b_i[3]*metric_RK4->A[i]       + b_i[4]*metric_RK5->A[i]);
                metric->B[i]        = metric->B[i]       + dt*(b_i[0]*metric_RK1->B[i]      + b_i[1]*metric_RK2->B[i]        + b_i[2]*metric_RK3->B[i]       + b_i[3]*metric_RK4->B[i]       + b_i[4]*metric_RK5->B[i]);
                metric->D_B[i]      = metric->D_B[i]     + dt*(b_i[0]*metric_RK1->D_B[i]    + b_i[1]*metric_RK2->D_B[i]      + b_i[2]*metric_RK3->D_B[i]     + b_i[3]*metric_RK4->D_B[i]     + b_i[4]*metric_RK5->D_B[i]);
                metric->U_tilda[i]  = metric->U_tilda[i] + dt*(b_i[0]*metric_RK1->U_tilda[i]+ b_i[1]*metric_RK2->U_tilda[i]  + b_i[2]*metric_RK3->U_tilda[i] + b_i[3]*metric_RK4->U_tilda[i] + b_i[4]*metric_RK5->U_tilda[i]);
                metric->K[i]        = metric->K[i]       + dt*(b_i[0]*metric_RK1->K[i]      + b_i[1]*metric_RK2->K[i]        + b_i[2]*metric_RK3->K[i]       + b_i[3]*metric_RK4->K[i]       + b_i[4]*metric_RK5->K[i]);
                metric->K_B[i]      = metric->K_B[i]     + dt*(b_i[0]*metric_RK1->K_B[i]    + b_i[1]*metric_RK2->K_B[i]      + b_i[2]*metric_RK3->K_B[i]     + b_i[3]*metric_RK4->K_B[i]     + b_i[4]*metric_RK5->K_B[i]);
                metric->lambda[i]   = metric->lambda[i]  + dt*(b_i[0]*metric_RK1->lambda[i] + b_i[1]*metric_RK2->lambda[i]   + b_i[2]*metric_RK3->lambda[i]  + b_i[3]*metric_RK4->lambda[i]  + b_i[4]*metric_RK5->lambda[i]);
                metric->alpha[i]    = metric->alpha[i]   + dt*(b_i[0]*metric_RK1->alpha[i]  + b_i[1]*metric_RK2->alpha[i]    + b_i[2]*metric_RK3->alpha[i]   + b_i[3]*metric_RK4->alpha[i]   + b_i[4]*metric_RK5->alpha[i]);
                metric->D_alpha[i]  = metric->D_alpha[i] + dt*(b_i[0]*metric_RK1->D_alpha[i]+ b_i[1]*metric_RK2->D_alpha[i]  + b_i[2]*metric_RK3->D_alpha[i] + b_i[3]*metric_RK4->D_alpha[i] + b_i[4]*metric_RK5->D_alpha[i]);


            }
            double rpoints[lattice_size];
            make_points(rpoints);
            double H1, H2, r_AH, B_AH, A, Am, Bf, Bfm, D_B, D_Bm, K_B, K_Bm, r, rm, dr;
            for (int i = 1; i < lattice_size; i++) {
                
                r  = rpoints[i];
                rm = rpoints[i - 1];
                dr = r - rm;

                A    = field_phys_i_sym (i,    B,    metric->A)+1.0;
                Am   = field_phys_i_sym (i-1,  B,    metric->A)+1.0;
                Bf   = field_phys_i_sym (i,    B,    metric->B)+1.0;
                Bfm  = field_phys_i_sym (i-1,  B,    metric->B)+1.0;
                D_B  = field_phys_i_asym(i,    B_as, metric->D_B);
                D_Bm = field_phys_i_asym(i-1,  B_as, metric->D_B);
                K_B  = field_phys_i_sym (i,    B,    metric->K_B);
                K_Bm = field_phys_i_sym (i-1,  B,    metric->K_B);

                H1 = 1.0 / sqrt(Am) * (2.0 / rm + D_Bm) - 2.0 * K_Bm;
                H2 = 1.0 / sqrt(A)  * (2.0 / r  + D_B)  - 2.0 * K_B;

                if (H1 < 0 && H2 > 0) {

                    r_AH = rm + dr * (-H1) / (H2 - H1);

                    B_AH = (r_AH - rm) / dr * (sqrt(Bf) - sqrt(Bfm)) + sqrt(Bfm);

                    apparent_horizon[n] = r_AH * B_AH;// sqrt(metric->B[i]);
                    printf("horizon is %.10f\n", r_AH* B_AH);
                    
                }
                
            }
            // save some things

            alpha_save[n] = field_phys_i_sym(0, B, metric->alpha)+1.0;
            
            if (n%step==0.0){
                for (int m=0; m<lattice_size; ++m){
                    field_save[k][m] = __real__ field_phys_i_sym_comp(m, B, q_fields[0]->phi[0][0]);// __real__(q_fields[0]->phi[0][0][m]);// field_phys_i_sym(m, B, c_fields->phi); //
                }
                k=k+1;
                printf("\n time is %d, and alpha at r=0  is: %.10f ", n, field_phys_i_sym(0, B, metric->alpha)+1.0);
            }

            hamiltonian_constraint(n+1, B, B_as, C_sym, C_asym, D, c_fields, metric, ham);
            
    }
    printf("\ndone");
    save_stress_tensor(1, cosm_const, c_fields, q_fields, metric, B, C_sym);
    ham_const_r(B, B_as, C_sym, C_asym, D, c_fields, metric, ham_r);

    //change to back to physical space
    dot_product_for_phys(B, c_fields->phi, c_fields_RK1->phi);
    dot_product_for_phys(B, c_fields->pi, c_fields_RK1->pi);

    dot_product_for_phys(B, metric->A, metric_RK1->A);
    dot_product_for_phys(B, metric->B, metric_RK1->B);
    dot_product_for_phys(B_as, metric->D_B, metric_RK1->D_B);
    dot_product_for_phys(B, metric->K, metric_RK1->K);
    dot_product_for_phys(B, metric->K_B, metric_RK1->K_B);
    dot_product_for_phys(B, metric->alpha, metric_RK1->alpha);
    dot_product_for_phys(B_as, metric->D_alpha, metric_RK1->D_alpha);
    dot_product_for_phys(B_as, metric->U_tilda, metric_RK1->U_tilda);
    dot_product_for_phys(B_as, metric->lambda, metric_RK1->lambda);
    #pragma omp parallel for
    for (int k = 0; k < number_of_k_modes; ++k) {
        for (int l = 0; l < number_of_l_modes; ++l) {
            int l_value;
            l_value = l_start + l * l_step;
            for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                dot_product_for_phys_comp(B, q_fields[which_q_field]->phi[k][l], q_fields_RK1[which_q_field]->phi[k][l]);
                dot_product_for_phys_comp(B, q_fields[which_q_field]->pi[k][l],  q_fields_RK1[which_q_field]->pi[k][l]);

            }
        }
    }
    save_field_comp(q_fields_RK1[0]->phi[0][0]);
    // again, just using c_fields_RK1 as place holder
    #pragma omp parallel for
    for (int i = 0; i < lattice_size; ++i) {
        c_fields->phi[i] = c_fields_RK1->phi[i];
        c_fields->pi[i] = c_fields_RK1->pi[i];

        metric->A[i] = metric_RK1->A[i]+1.0;
        metric->B[i] = metric_RK1->B[i]+1.0;
        metric->D_B[i] = metric_RK1->D_B[i];
        metric->K[i] = metric_RK1->K[i];
        metric->K_B[i] = metric_RK1->K_B[i];
        metric->alpha[i] = metric_RK1->alpha[i]+1.0;
        metric->D_alpha[i] = metric_RK1->D_alpha[i];
        metric->U_tilda[i] = metric_RK1->U_tilda[i];
        metric->lambda[i] = metric_RK1->lambda[i];

        c_fields_RK1->phi[i] = 0.0;
        c_fields_RK1->pi[i] = 0.0;

        metric_RK1->A[i] = 0.0;
        metric_RK1->B[i] = 0.0;
        metric_RK1->D_B[i] = 0.0;
        metric_RK1->K[i] = 0.0;
        metric_RK1->K_B[i] = 0.0;
        metric_RK1->alpha[i] = 0.0;
        metric_RK1->D_alpha[i] = 0.0;
        metric_RK1->U_tilda[i] = 0.0;
        metric_RK1->lambda[i] = 0.0;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that frees up the memory */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void free_memory(Classical_Fields* c_fields, Quantum_Fields** q_fields, Metric_Fields* metric) {

    free(metric->A);
    free(metric->B);
    free(metric->D_B);
    free(metric->U_tilda);
    free(metric->K);
    free(metric->K_B);
    free(metric->lambda);
    free(metric->alpha);
    free(metric->D_alpha);
    free(metric);

    free(c_fields->phi);
    free(c_fields->pi);
    free(c_fields);
    
    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(q_fields[which_q_field]->phi[k][l]);
                free(q_fields[which_q_field]->pi[k][l]);
            }
            free(q_fields[which_q_field]->phi[k]);
            free(q_fields[which_q_field]->pi[k]);
        }
        free(q_fields[which_q_field]->phi);
        free(q_fields[which_q_field]->pi);

        free(q_fields[which_q_field]);
    }
    free(q_fields);


}
/////////////////////////////////////////////

/* Main */

void main(){
    double timestart = omp_get_wtime();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* DEFINE VARIABLES AND ALLOCATE THEIR MEMORY*/
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double   a_ij[nu_legendre][nu_legendre];             //coefficients of the Runge-Kutta evolution
    double   b_i[nu_legendre];                           //coefficients of the Runge-Kutta evolution
    double   c_i[nu_legendre];                           //the zeros of P_nu(2c - 1)=0, i.e. P_nu(2c_1 - 1)
    double   GL_matrix_inverse[nu_legendre][nu_legendre];         //this comes from GL_matrix*(a_ij)=(c_i^l) for (a_ij)
                                                                                     //                          (b_i)  (1/l  )     (b_i )

    find_c_i(c_i);
    find_GL_matrix_inverse(c_i, GL_matrix_inverse);
    find_a_ij__b_i(c_i, b_i, a_ij, GL_matrix_inverse);

    
    double** B;
    B = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        B[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** B_pi;
    B_pi = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        B_pi[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** B_as;
    B_as = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        B_as[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** B_inv_sym;
    B_inv_sym = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        B_inv_sym[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** B_inv_asym;
    B_inv_asym = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        B_inv_asym[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** C_sym;
    C_sym = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        C_sym[i] = (double*)malloc(lattice_size * sizeof(double));
    }
    double** C_asym;
    C_asym = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        C_asym[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** D;
    D = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        D[i] = (double*)malloc(lattice_size * sizeof(double));
    }


    Metric_Fields *metric;
    Metric_Fields *metric_RK1;
    Metric_Fields *metric_RK2;
    Metric_Fields *metric_RK3;
    Metric_Fields *metric_RK4;
    Metric_Fields *metric_RK5;
    Metric_Fields *metric_RK_sum;

    metric                  = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric->A               = (double*)malloc(lattice_size*sizeof(double));
    metric->B               = (double*)malloc(lattice_size*sizeof(double));
    metric->D_B             = (double*)malloc(lattice_size*sizeof(double));
    metric->U_tilda         = (double*)malloc(lattice_size*sizeof(double));
    metric->K               = (double*)malloc(lattice_size*sizeof(double));
    metric->K_B             = (double*)malloc(lattice_size*sizeof(double));
    metric->lambda          = (double*)malloc(lattice_size*sizeof(double));
    metric->alpha           = (double*)malloc(lattice_size*sizeof(double));
    metric->D_alpha         = (double*)malloc(lattice_size*sizeof(double));

    metric_RK1              = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK1->A           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->B           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->D_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->U_tilda     = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->K           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->K_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->lambda      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->alpha       = (double*)malloc(lattice_size*sizeof(double));
    metric_RK1->D_alpha     = (double*)malloc(lattice_size*sizeof(double));

    metric_RK2              = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK2->A           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->B           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->D_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->U_tilda     = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->K           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->K_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->lambda      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->alpha       = (double*)malloc(lattice_size*sizeof(double));
    metric_RK2->D_alpha     = (double*)malloc(lattice_size*sizeof(double));

    metric_RK3              = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK3->A           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->B           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->D_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->U_tilda     = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->K           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->K_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->lambda      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->alpha       = (double*)malloc(lattice_size*sizeof(double));
    metric_RK3->D_alpha     = (double*)malloc(lattice_size*sizeof(double));

    metric_RK4              = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK4->A           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->B           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->D_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->U_tilda     = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->K           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->K_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->lambda      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->alpha       = (double*)malloc(lattice_size*sizeof(double));
    metric_RK4->D_alpha     = (double*)malloc(lattice_size*sizeof(double));

    metric_RK5              = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK5->A           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->B           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->D_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->U_tilda     = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->K           = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->K_B         = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->lambda      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->alpha       = (double*)malloc(lattice_size*sizeof(double));
    metric_RK5->D_alpha     = (double*)malloc(lattice_size*sizeof(double));

    metric_RK_sum           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK_sum->A        = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->B        = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->D_B      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->U_tilda  = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->K        = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->K_B      = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->lambda   = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->alpha    = (double*)malloc(lattice_size*sizeof(double));
    metric_RK_sum->D_alpha  = (double*)malloc(lattice_size*sizeof(double));


    Classical_Fields *c_fields;
    Classical_Fields *c_fields_RK1;
    Classical_Fields *c_fields_RK2;
    Classical_Fields *c_fields_RK3;
    Classical_Fields *c_fields_RK4;
    Classical_Fields *c_fields_RK5;
    Classical_Fields *c_fields_RK_sum;

    Quantum_Fields** q_fields;
    Quantum_Fields** q_fields_RK1;
    Quantum_Fields** q_fields_RK2;
    Quantum_Fields** q_fields_RK3;
    Quantum_Fields** q_fields_RK4;
    Quantum_Fields** q_fields_RK5;
    Quantum_Fields** q_fields_RK_sum;

    c_fields             = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields->phi        = (double*)malloc(lattice_size * sizeof(double));
    c_fields->pi         = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK1         = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK1->phi    = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK1->pi     = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK2         = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK2->phi    = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK2->pi     = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK3         = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK3->phi    = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK3->pi     = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK4         = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK4->phi    = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK4->pi     = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK5         = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK5->phi    = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK5->pi     = (double*)malloc(lattice_size * sizeof(double));

    c_fields_RK_sum      = (Classical_Fields*)malloc(sizeof(Classical_Fields));
    c_fields_RK_sum->phi = (double*)malloc(lattice_size * sizeof(double));
    c_fields_RK_sum->pi  = (double*)malloc(lattice_size * sizeof(double));
    
    //allocate memory for quantum modes, accessed with q_fields[which_q_field]->phi_mode[k][l][i]
    q_fields            = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK1        = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK2        = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK3        = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK4        = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK5        = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));
    q_fields_RK_sum     = (Quantum_Fields **)malloc(number_of_q_fields * sizeof(Quantum_Fields*));

    for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

        q_fields[which_q_field]         = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK1[which_q_field]     = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK2[which_q_field]     = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK3[which_q_field]     = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK4[which_q_field]     = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK5[which_q_field]     = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));
        q_fields_RK_sum[which_q_field]  = (Quantum_Fields *)malloc(sizeof(Quantum_Fields));

        q_fields[which_q_field]->phi       = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields[which_q_field]->pi        = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK1[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK1[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK2[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK2[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK3[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK3[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK4[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK4[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK5[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK5[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK_sum[which_q_field]->phi= (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK_sum[which_q_field]->pi = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        for (int k=0; k<number_of_k_modes; k++){

            q_fields[which_q_field]->phi[k]         = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields[which_q_field]->pi[k]          = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK1[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK1[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK2[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK2[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK3[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK3[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK4[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK4[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK5[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK5[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK_sum[which_q_field]->phi[k]  = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK_sum[which_q_field]->pi[k]   = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            for(int l=0;l<number_of_l_modes;++l){

                q_fields[which_q_field]->phi[k][l]         = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields[which_q_field]->pi[k][l]          = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK1[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK1[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK2[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK2[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK3[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK3[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK4[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK4[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK5[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK5[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));

                q_fields_RK_sum[which_q_field]->phi[k][l]  = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
                q_fields_RK_sum[which_q_field]->pi[k][l]   = (__complex__ double *)malloc(lattice_size * sizeof(__complex__ double));
            }
        }
    }

    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* INITIALISE CALCULATIONS */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    save_points();

    find_B_sym(B);
    find_B_asym(B_as);
    find_C_sym(C_sym);
    find_C_asym(C_asym);
    find_D_sym(D);
    find_B_inverse_sym(B,B_inv_sym);
    find_B_inverse_asym(B_as,B_inv_asym);
    find_B_sym(B);
    find_B_asym(B_as);
    double rpoints[lattice_size];
    double cosm_const;
    make_points(rpoints);
    save_points(rpoints);

    printf("Number of time steps will be %d,\n", evolve_time_int);
    initial_conditions_quantum(c_fields, q_fields, metric);

    cosm_const = set_cosm_constant(c_fields, q_fields, metric, B, C_sym);
    printf("The cosmological constant is %.10f,\n", cosm_const);

    initial_conditions_classical(c_fields, metric);
    initialise                  (c_fields, metric, B, B_as,  B_inv_sym, B_inv_asym);
    
    save_field(c_fields->phi);
    

    double ham_r[lattice_size];
    double field_save[(evolve_time_int/step)+1][lattice_size];
    double alpha_save[evolve_time_int], apparent_horizon[evolve_time_int], ham[evolve_time_int];
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* ACTUAL EVOLUTION */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    fullEvolution(B, B_as, B_inv_sym, B_inv_asym, C_sym, C_asym, D, b_i, a_ij, c_fields_RK1, c_fields_RK2, c_fields_RK3, c_fields_RK4, c_fields_RK5, c_fields_RK_sum, c_fields,
                                                                               q_fields_RK1, q_fields_RK2, q_fields_RK3, q_fields_RK4, q_fields_RK5, q_fields_RK_sum, q_fields,
                                                                               metric_RK1,   metric_RK2,   metric_RK3,   metric_RK4,   metric_RK5,   metric_RK_sum,   metric, 
                                                  field_save, alpha_save, apparent_horizon, ham, ham_r, cosm_const);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* SAVE THINGS */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    save_field_comp(q_fields[0]->phi[0][0]);
    save_field1(ham_r);
    save_field2(metric->K_B);
    save_alpha_t(alpha_save);
    save_hor_t(apparent_horizon);
    save_ham_t(ham);
       
    FILE* finout;
    finout = fopen("scalar_evolution_matrix.txt", "w");
    for (int n = 0; n < (int)(evolve_time_int / step) + 1; ++n) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size; ++m) {
            fprintf(finout, "%.200f ", (field_save[n][m]));
        }
    }
    fclose(finout);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* FREE ALL THE MEMORY */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (int i = 0; i < lattice_size; i++) {
        free(B[i]);
    }
    free(B);

    for (int i = 0; i < lattice_size; i++) {
        free(B_pi[i]);
    }
    free(B_pi);

    for (int i = 0; i < lattice_size; i++) {
        free(B_as[i]);
    }
    free(B_as);

    for (int i = 0; i < lattice_size; i++) {
        free(B_inv_sym[i]);
    }
    free(B_inv_sym);


    for (int i = 0; i < lattice_size; i++) {
        free(B_inv_asym[i]);
    }
    free(B_inv_asym);


    for (int i = 0; i < lattice_size; i++) {
        free(C_sym[i]);
    }
    free(C_sym);

    for (int i = 0; i < lattice_size; i++) {
        free(C_asym[i]);
    }
    free(C_asym);

    for (int i = 0; i < lattice_size; i++) {
        free(D[i]);
    }
    free(D);

    
    free_memory(c_fields,        q_fields,        metric);
    free_memory(c_fields_RK1,    q_fields_RK1,    metric_RK1);
    free_memory(c_fields_RK2,    q_fields_RK2,    metric_RK2);
    free_memory(c_fields_RK3,    q_fields_RK3,    metric_RK3);
    free_memory(c_fields_RK4,    q_fields_RK4,    metric_RK4);
    free_memory(c_fields_RK5,    q_fields_RK5,    metric_RK5);
    free_memory(c_fields_RK_sum, q_fields_RK_sum, metric_RK_sum);

    printf("The code took %f", omp_get_wtime() - timestart);

}
