#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "matrix_operations.h"

/* 1 - A */
/* Univariate gaussian data generator, outcome ~ N(mean, var) */
double Marsaglia_polar_data_generator(double mean, double var)
{
    double x, y, s;
    do
    {
        x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        y = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = pow(x, 2) + pow(y, 2);
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    return mean + sqrt(var) * x * s;
};

/* 1 - B */
/* Polynomial basis linear model data generator, outcome ~  */
typedef struct data_point
{
    double x;
    double y;
} data_point;
/* y = WTPhi(x)+e ; e ~ N(0, a) */
/* w0, w1, w2, ... */
data_point Polynomial_linear_data_generator(int num_of_basis, double a, double *w)
{
    int i = 0;
    double y = 0;
    /* an internal constraint: -10.0 < x < 10.0, x is uniformly distributed. */
    double x = (rand() / (double)RAND_MAX) * 20.0 - 10.0, x_poly = 1;
    for (i = 0; i < num_of_basis; i++)
    {
        y += x_poly * w[i];
        x_poly *= x;
    }

    data_point data;
    data.x = x;
    data.y = y + Marsaglia_polar_data_generator(0, a);

    return data;
};

/* 2 - Sequential estimate */
/* Welford's Online algorithm */
int Welford_sequential_estimate(double data_x)
{
    static double old_mean = 0, old_var = 0, new_mean = 0, new_var = 0;
    static int num_of_data;

    /* new data come in */
    num_of_data++;

    /* new mean */
    new_mean = old_mean + (data_x - old_mean) / num_of_data;
    /* new variance */
    new_var = old_var + ((data_x - new_mean) * (data_x - old_mean) - old_var) / num_of_data;
    printf("data: %f, m: %f, s: %f\n", data_x, new_mean, new_var);
    if (fabs(old_mean - new_mean) < 0.00001 && fabs(old_var - new_var) < 0.0001)
    {
        return 1;
    }
    old_mean = new_mean;
    old_var = new_var;

    return 0;
};

/* Part 3 */
typedef struct Gaussian_distribution
{
    double *mean;
    double var;
} Gaussian_distribution;

/* Bayesian Linear Regression - training per data */
int Bayesian_Linear_Regression_one(Gaussian_distribution *prior, data_point *data, int *num_of_basis, double *random_error_precision)
{
    /* Phi */
    double *Phi = (double *)malloc(sizeof(double) * *num_of_basis);
    int i = 0;
    Phi[0] = 1.0;
    printf("Phi: %f ", Phi[0]);
    for (i = 1; i < *num_of_basis; i++)
    {
        Phi[i] = Phi[i - 1] * data->x;
        printf("%f ", Phi[i]);
    }
    printf("\n");

    /* predictive distribution */
    double mean_predictive = 0, var_predictive = 0, PhiTPhi = 0;
    for (i = 0; i < *num_of_basis; i++)
    {
        mean_predictive += prior->mean[i] * Phi[i];
        PhiTPhi += pow(Phi[i], 2);
        var_predictive += prior->var * pow(Phi[i], 2);
    }

    var_predictive += *random_error_precision;
    var_predictive = 1 / var_predictive;
    printf("Predictive distribution's mean: %f, var: %f\n", mean_predictive, var_predictive);

    /* update prior */
    double last_prior_var = 0, last_prior_mean = 0;
    last_prior_var = prior->var;
    prior->var = 1 / prior->var + PhiTPhi / (*random_error_precision);
    prior->var = 1 / prior->var;
    printf("Update Prior's mean: ");
    for (i = 0; i < *num_of_basis; i++)
    {
        last_prior_mean = prior->mean[i];
        prior->mean[i] = prior->var * (data->y * Phi[i] / (*random_error_precision) + prior->mean[i] / last_prior_var);
        //eps += fabs(prior->mean[i] - last_prior_mean);
        printf("%f ", prior->mean[i]);
    }
    free(Phi);
    printf(", var: %f\n\n", prior->var);

    if (prior->var < 0.0001)
    {
        printf("Converge!\n");
        return 1;
    }
    return 0;
};

/* Bayesian Linear Regression - training per # of data */
typedef struct Multiv_Gaussian_distribution
{
    matrix *mean;
    matrix *var;
} Multiv_Gaussian_distribution;

int Bayesian_Linear_Regression_batch(Multiv_Gaussian_distribution *prior, Multiv_Gaussian_distribution *posterior, data_point data[], int *num_of_basis, double *random_error_precision)
{
    /* Phi */
    matrix *Phi = create_matrix(3, *num_of_basis);
    int i = 0, j = 0;
    Phi->data[0][0] = 1.0;
    Phi->data[1][0] = 1.0;
    Phi->data[2][0] = 1.0;
    printf("Phi:\n");
    for (i = 0; i < 3; i++)
    {
        for (j = 1; j < *num_of_basis; j++)
        {
            Phi->data[i][j] = Phi->data[i][j - 1] * data[i].x;
        }
    }
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < *num_of_basis; j++)
        {
            printf("%f ", Phi->data[i][j]);
        }
        printf("\n");
    }

    /* predictive distribution */
    matrix *Phi_T = tran_matrix(Phi);
    //print_matrix(Phi_T);
    matrix *mean_predictive = multi_matrix(Phi, prior->mean);
    matrix *Phi_S = multi_matrix(Phi, prior->var);
    //print_matrix(Phi_S);
    matrix *var_predictive = multi_matrix(Phi_S, Phi_T);
    for (i = 0; i < var_predictive->dimensions[1]; i++)
    {
        var_predictive->data[i][i] += 1 / (*random_error_precision);
    }
    printf("Predictive distribution's mean:\n");
    for (i = 0; i < mean_predictive->dimensions[0]; i++)
    {
        for (j = 0; j < mean_predictive->dimensions[1]; j++)
        {
            printf("%f ", mean_predictive->data[i][j]);
        }
        printf("\n");
    }
    printf("var:\n");
    for (i = 0; i < var_predictive->dimensions[0]; i++)
    {
        for (j = 0; j < var_predictive->dimensions[1]; j++)
        {
            printf("%f ", var_predictive->data[i][j]);
        }
        printf("\n");
    }

    /* previous prior */
    /*
    printf("Prior\nmean:\n");
    for (i = 0; i < prior->mean->dimensions[0]; i++)
    {
        for (j = 0; j < prior->mean->dimensions[1]; j++)
        {
            printf("%f ", prior->mean->data[i][j]);
        }
        printf("\n");
    }
    printf("var:\n");
    for (i = 0; i < prior->var->dimensions[0]; i++)
    {
        for (j = 0; j < prior->var->dimensions[1]; j++)
        {
            printf("%f ", prior->var->data[i][j]);
        }
        printf("\n");
    }
    */

    /* update prior */
    /* var */
    printf("Update Prior's var: \n");
    matrix *S0_inverse = inverse(prior->var);
    matrix *PhiTPhi = multi_matrix(Phi_T, Phi);
    for (i = 0; i < PhiTPhi->dimensions[0]; i++)
    {
        for (j = 0; j < PhiTPhi->dimensions[0]; j++)
        {
            PhiTPhi->data[i][j] *= *random_error_precision;
        }
    }
    matrix *new_var_inverse = add_matrix(S0_inverse, PhiTPhi);
    posterior->var = inverse(new_var_inverse);
    for (i = 0; i < posterior->var->dimensions[0]; i++)
    {
        for (j = 0; j < posterior->var->dimensions[1]; j++)
        {
            printf("%f ", posterior->var->data[i][j]);
        }
        printf("\n");
    }
    /* mean */
    printf("Update Prior's mean: \n");
    matrix *y = create_matrix(3, 1);
    y->data[0][0] = data[0].y;
    y->data[1][0] = data[1].y;
    y->data[2][0] = data[2].y;
    matrix *PhiTy = multi_matrix(Phi_T, y);
    for (i = 0; i < PhiTy->dimensions[0]; i++)
    {
        PhiTy->data[i][0] *= *random_error_precision;
    }
    matrix *S0_inverse_mean0 = multi_matrix(S0_inverse, prior->mean);
    matrix *PhiTy_add_S0_inverse_mean0 = add_matrix(PhiTy, S0_inverse_mean0);
    posterior->mean = multi_matrix(posterior->var, PhiTy_add_S0_inverse_mean0);

    for (i = 0; i < posterior->mean->dimensions[0]; i++)
    {
        for (j = 0; j < posterior->mean->dimensions[1]; j++)
        {
            printf("%f ", posterior->mean->data[i][j]);
        }
        printf("\n");
    }

    double eps = 0;
    for (i = 0; i < prior->var->dimensions[0]; i++)
    {
        for (j = 0; j < prior->var->dimensions[1]; j++)
        {
            eps += fabs(prior->var->data[i][j] - posterior->var->data[i][j]);
        }
    }
    assign_matrix(posterior->mean, prior->mean, 0, true);
    assign_matrix(posterior->var, prior->var, 0, true);
    eps /= (posterior->var->dimensions[0] * posterior->var->dimensions[1]);
    printf("Error: %f\n", eps);

    free_mat(Phi);
    free_mat(Phi_T);
    free_mat(mean_predictive);
    free_mat(Phi_S);
    free_mat(var_predictive);
    free_mat(S0_inverse);
    free_mat(PhiTPhi);
    free_mat(new_var_inverse);
    free_mat(y);
    free_mat(PhiTy);
    free_mat(S0_inverse_mean0);

    if (eps < 0.0000001)
    {
        printf("Converge!\n");
        return 1;
    }
    return 0;
};

/* output data for plot */
int output(data_point *past_data, int *iter, Gaussian_distribution *prior, int *num_of_basis)
{
    FILE *fp;
    double y = 0, phi = 1;

    fp = fopen("plot.csv", "a+");

    if (!fp)
        return -1;
    fprintf(fp, "%d\n", *iter);
    for (int i = 0; i < *iter; i++)
    {
        y = prior->mean[0];
        phi = 1;
        for (int j = 1; j < *num_of_basis; j++)
        {
            phi *= past_data[i].x;
            y += prior->mean[j] * phi;
        }
        /* see Gaussian distribution of each data point */
        fprintf(fp, "%f, %f, %f, %f, %f, %f, %f, %f, %f\n", (past_data + i)->x, (past_data + i)->y, y,
                y + prior->var, y + prior->var * 2, y + prior->var * 3,
                y - prior->var, y - prior->var * 2, y - prior->var * 3);
    }
    fclose(fp);
    return 0;
};

int main(int argc, char **argv)
{
    srand(time(NULL));
    /* 1 - m, s */
    /* 2 - # of basis, a, w1, w2, ... */
    if (atoi(argv[1]) == 2)
    {
        double w[2] = {1, 2};
        data_point a;
        for (int i = 0; i < 10; i++)
        {
            a = Polynomial_linear_data_generator(2, 1, w);
            printf("%f, %f\n", a.x, a.y);
        }
    }
    /* 3 - m, s */
    if (atoi(argv[1]) == 3)
    {
        int i = 0, converge = 0;
        double input_data_x = 0;
        while (true)
        {
            i++;
            printf("iteration %d: ", i);
            input_data_x = Marsaglia_polar_data_generator(atof(argv[2]), atof(argv[3]));
            //printf("%f\n", input_data_x);
            if (Welford_sequential_estimate(input_data_x) == 1 && i > 1)
                break;
        }
    }
    /* Bayesian Linear Regression - training per data */
    else if (atoi(argv[1]) == 4)
    {
        data_point input;
        int i = 0, converge = 0, iter = 0;
        /* handle num_of_basis, precision, w */
        int num_of_basis = atoi(argv[2]);
        /* likelihood's variance / random_error_precision, aI*/
        double random_error_precision = atof(argv[3]);
        double *w_truth = (double *)malloc(sizeof(double) * num_of_basis);

        /* data storage */
        data_point past_data[1000] = {};

        /* last one is precision for initial prior */
        Gaussian_distribution prior;
        prior.mean = (double *)malloc(sizeof(double) * num_of_basis);
        printf("target function's parameters: a is %f; weights: ", random_error_precision);
        for (i = 0; i < num_of_basis; i++)
        {
            w_truth[i] = atof(argv[4 + i]);
            printf("%f ", w_truth[i]);
            prior.mean[i] = 0.0;
        }
        printf("\n");
        prior.var = 1 / (atof(argv[4 + i]));
        double guess_error_precision = atof(argv[5 + i]);
        printf("Guess error_precision: %f\n", guess_error_precision);
        printf("initial prior N([ ");
        for (i = 0; i < num_of_basis; i++)
        {
            printf("%f ", prior.mean[i]);
        }
        printf("], %f)\n\n", prior.var);

        do
        {
            /* generate data point */
            input = Polynomial_linear_data_generator(num_of_basis, random_error_precision, w_truth);
            past_data[iter] = input;
            iter++;
            printf("Iteration : %d\nInput x: %f, y: %f\n", iter, input.x, input.y);
            /* feed prior, input, num_of_basis & a (likelihood's variance) into Bayesian Linear Regression */
            converge = Bayesian_Linear_Regression_one(&prior, &input, &num_of_basis, &random_error_precision);

            /* Plot!! */
            output(past_data, &iter, &prior, &num_of_basis);

        } while (converge == 0);
        free(prior.mean);
        free(w_truth);
        printf("Check the prosterior capability\n");
    }
    /* Bayesian Linear Regression - training per # of data */
    else if (atoi(argv[1]) == 5)
    {
        data_point input[3];
        int i = 0, j = 0, converge = 0, iter = 0;
        /* handle num_of_basis, precision, w */
        int num_of_basis = atoi(argv[2]);
        printf("# of basis is %d\n", num_of_basis);
        /* likelihood's variance / random_error_precision, aI*/
        double random_error_precision = atof(argv[3]);
        matrix *w_truth = create_matrix(1, num_of_basis);
        printf("Likelihood's var is %f*I\n", 1 / random_error_precision);

        /* data storage */
        //data_point past_data[1000][3] = {};

        /* last one is precision for initial prior */
        Multiv_Gaussian_distribution prior;
        Multiv_Gaussian_distribution posterior;
        prior.mean = create_matrix(num_of_basis, 1);
        prior.var = create_matrix(num_of_basis, num_of_basis);
        printf("Weights: ");
        for (i = 0; i < num_of_basis; i++)
        {
            w_truth->data[0][i] = atof(argv[4 + i]);
            printf("%f ", w_truth->data[0][i]);
            prior.mean->data[i][0] = 0.0;
            prior.var->data[i][i] = 1 / (atof(argv[4 + num_of_basis]));
        }
        printf("\n");
        //double guess_error_precision = atof(argv[5 + i]);
        //printf("Guess error_precision: %f\n", guess_error_precision);
        printf("initial prior N([ ");
        for (i = 0; i < num_of_basis; i++)
        {
            printf("%f ", prior.mean->data[i][0]);
        }
        printf("]\ncovar is\n");
        for (i = 0; i < num_of_basis; i++)
        {
            for (j = 0; j < num_of_basis; j++)
                printf("%f ", prior.var->data[i][j]);
            printf("\n");
        }
        printf("\n\n");

        do
        {
            /* generate data point */
            input[0] = Polynomial_linear_data_generator(num_of_basis, 1 / random_error_precision, w_truth->data[0]);
            input[1] = Polynomial_linear_data_generator(num_of_basis, 1 / random_error_precision, w_truth->data[0]);
            input[2] = Polynomial_linear_data_generator(num_of_basis, 1 / random_error_precision, w_truth->data[0]);
            //past_data[iter][0] = input[0];
            //past_data[iter][1] = input[1];
            //past_data[iter][2] = input[2];
            iter++;
            printf("\n\nIteration : %d\n", iter);
            printf("Input x: %f, y: %f\n", input[0].x, input[0].y);
            printf("Input x: %f, y: %f\n", input[1].x, input[1].y);
            printf("Input x: %f, y: %f\n", input[2].x, input[2].y);
            /* feed prior, input, num_of_basis & a (likelihood's variance) into Bayesian Linear Regression */
            converge = Bayesian_Linear_Regression_batch(&prior, &posterior, input, &num_of_basis, &random_error_precision);

            /* Plot!! */
            //output(past_data, &iter, &prior, &num_of_basis);

        } while (/*iter < 2*/ converge == 0);

        free_mat(prior.mean);
        free_mat(prior.var);
        free_mat(posterior.mean);
        free_mat(posterior.var);
        free_mat(w_truth);
        printf("Check the prosterior capability\n");
    }
    return 0;
}