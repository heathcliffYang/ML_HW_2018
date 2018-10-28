#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

double eps_threshold = 0.000001;

using namespace std;

bool x_not_converge(double *x_n, double *x_n_1, int num_of_poly_basis) {
    double eps = 0;
    for (int i=0; i<num_of_poly_basis; i++) {
        eps += abs(x_n[i] - x_n_1[i]);
    }
    if (eps > eps_threshold) {
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            x_n_1[i] = x_n[i];
        }
        return true;
    }
    return false;
}

bool dx_larger_than_threshold(double *dx, double *x_n, int num_of_poly_basis)
{
    double eps = 0;
    for (int i = 0; i < num_of_poly_basis; i++)
    {
        eps += abs(dx[i]);
    }
    if (eps > eps_threshold)
    {
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            x_n[i] += dx[i];
        }
        return true;
    }
    return false;
}



int main(int argc, char **argv)
{
    /* Combine file path and name */
    char *filepath = strcat(argv[1], argv[2]);

    printf("Data input file is \"%s\"\n", filepath);

    /* polynomial of degree n */
    int num_of_poly_basis =  atoi(argv[3]);

    printf("Basis function is polynomials of degree %d\n", num_of_poly_basis);

    /* lambda for regularization */
    double lambda = atof(argv[4]);  /////

    printf("Lambda for regularization is %f\n", lambda);

    FILE *fp;
    char buf[255];
    /* Assume polynomial degree will be smaller than 10 */
    int num_of_input_data=0;
    double A[1000][10], t[1000];

    fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("Fail to open the input file\n");
        return 0;
    }

    /* Make A matrix       x0^n + x0^(n-1) + ... + x0^1 + 1 */
    /* num_of_input_data { x1^n + x1^(n-1) + ... + x1^1 + 1 */
    /*                             num_of_poly_basis        */
    while (fgets(buf, 255, fp)) {
        char *num_1_ptr = strtok(buf, ",");
        char *num_2_ptr = strtok(NULL, "\n");

        A[num_of_input_data][0] = 1;
        A[num_of_input_data][1] = atof(num_1_ptr);
        t[num_of_input_data] = atof(num_2_ptr);

        for (int j=2; j<num_of_poly_basis; j++) {
            A[num_of_input_data][j] = A[num_of_input_data][j-1] * A[num_of_input_data][1];
        }

        /* Just check if my input is correct or not */
        // printf("My data input x,y is (%d, %d)\n", A[num_of_input_data][1], t[num_of_input_data]);
        num_of_input_data++;
    }

    /* Check how the A matrix looks like */
    printf("\nA is :\n");
    for (int i=0; i<num_of_input_data; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            printf("%4.2f ", A[i][j]);
        }
        printf("\n");
    }


    /* a. For LSE */
    /* 1. Use LU decomposition to find the inverse of (ATA + lambda*I) */
    /* AT */
    double AT[10][1000];
    for (int i=0; i<num_of_input_data; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            AT[j][i] = A[i][j];
        }
    }
    /* ATA + lambda*I */
    double ATA_for_Newton[10][10] = {}, ATA_for_Newton_orig[10][10] = {};
    double ATA[10][10] = {};
    for (int i=0; i<num_of_poly_basis; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            for (int k=0; k<num_of_input_data; k++) {
                ATA[i][j] +=  AT[i][k] * A[k][j];
                ATA_for_Newton[i][j] = ATA[i][j];
                ATA_for_Newton_orig[i][j] = ATA[i][j];
            }
            if (i==j)
                ATA[i][j] += lambda;
        }
    }

    printf("\nATA + lambda*I is :\n");
    for (int i=0; i<num_of_poly_basis; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            printf("%.2f ", ATA[i][j]);
        }
        printf("\n");
    }

    /* LU decomposition */
    double U[10][10]={}, L[10][10]={};
    for (int k=0; k<num_of_poly_basis; k++) {
        for (int j=k; j<num_of_poly_basis; j++) {
            U[k][j] = ATA[k][j];
            for (int s=0; s<=k-1; s++)
                U[k][j] -= L[k][s]*U[s][j]; 
        }

        L[k][k] = 1;
        for (int i=k+1; i<num_of_poly_basis; i++) {
            L[i][k] = ATA[i][k];
            for (int s=0; s<=k-1; s++)
                L[i][k] -= L[i][s]*U[s][k];
            L[i][k] /= U[k][k];
        }
    }

    printf("\nL is :\n");

    for (int i=0; i<num_of_poly_basis; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            printf("%f ", L[i][j]);
        }
        printf("\n");
    }

    printf("\nU is :\n");

    for (int i=0; i<num_of_poly_basis; i++) {
        for (int j=0; j<num_of_poly_basis; j++) {
            printf("%.2f ", U[i][j]);
        }
        printf("\n");
    }

    /* Ly = ATb */
    double Y[1000] = {};
    for (int i=0; i<num_of_poly_basis; i++) {
        for (int j = 0; j < num_of_input_data; j++) {
            Y[i] += AT[i][j]*t[j];
        }
        for (int k=0; k<i; k++) {
            Y[i] -= L[i][k]*Y[k];
        }
    }
    
    /* Ux = y */
    double x[10]={};
    for (int i=num_of_poly_basis-1; i>=0; i--) {
        x[i] = Y[i];
        for (int j=num_of_poly_basis-1; j>i; j--) {
            x[i] -= x[j]*U[i][j];
        }
        x[i] /=U[i][i];
    }

    printf("\nx is :\n");
    for (int i=0; i < num_of_poly_basis-1; i++)
        printf("%f ", x[i]);
    printf("%f\n", x[num_of_poly_basis-1]);

    /* 2. Print out the equation of the best fitting line and the error. */
    printf("Eq is y = %f", x[0]);
    for (int i=1; i<num_of_poly_basis; i++) {
        printf(" + %f*x^%d", x[i], i);
    }
    double y[1000] = {};
    double Er = 0, Er_rmse = 0;
    for (int i = 0; i < num_of_input_data; i++)
    {
        for (int j = 0; j < num_of_poly_basis; j++)
        {
            y[i] += A[i][j] * x[j];
        }
        Er += pow(abs(y[i] - t[i]), 2);
    }
    printf("\nLeast Square Error is %f\nRMSE is %f\n", Er, sqrt(Er/num_of_input_data));

    /* b. For Newton's method */
    /* 1. Print out the equation of the best fitting line and the error, and compare to LSE. */
    /* gradient x_n = x_n-1 - (ATA_for_Newton)^-1 (ATA_for_Newton x_n-1 - AT b) */
    double x_for_Newton_n[10]={}, x_for_Newton_n_1[10]={}, Udx[10]={}, dx[10]={};
    int iter = -1;

    /* Find the inverse of ATA_for_Newton */
    double ATA_inverse[10][10] = {}, op = 0;
    for (int i = 0; i < num_of_poly_basis; i++)
    {
        op = ATA_for_Newton[i][i];
        ATA_inverse[i][i] += 1 / op;
        for (int j = 0; j < num_of_poly_basis; j++)
        {
            if (j != i)
            {
                ATA_for_Newton[i][j] /= op;
                ATA_inverse[i][j] /= op;
            }
        }
        for (int j = 0; j < num_of_poly_basis; j++)
        {
            if (j != i)
            {
                op = -ATA_for_Newton[j][i];
                for (int k = 0; k < num_of_poly_basis; k++)
                {
                    ATA_for_Newton[j][k] += ATA_for_Newton[i][k] * op;
                    ATA_inverse[j][k] += ATA_inverse[i][k] * op;
                }
            }
        }
    }

    double ATb[10] = {}, tmp_x[10] = {};
    for (int i=0; i<num_of_poly_basis; i++) {
        for (int k = 0; k < num_of_input_data; k++)
        {
            ATb[i] += AT[i][k] * t[k];
        }
    }

    do {
        iter ++;

        for (int i = 0; i < num_of_poly_basis; i++)
        {
            tmp_x[i] = 0;
            x_for_Newton_n[i] = 0;
        }

        /* ATA_for_Newton x_n - 1 */
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            for (int j = 0; j < num_of_poly_basis; j++) {
                tmp_x[i] += ATA_for_Newton_orig[i][j]*x_for_Newton_n_1[j];
            }
            tmp_x[i] -= ATb[i];
        }

        /* H * x'*/
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            for (int j = 0; j < num_of_poly_basis; j++)
            {
                x_for_Newton_n[i] -= ATA_inverse[i][j]*tmp_x[j];
            }
            x_for_Newton_n[i] += x_for_Newton_n_1[i];
        }

    } while (x_not_converge(x_for_Newton_n, x_for_Newton_n_1, num_of_poly_basis));

    printf("\nNewton's method version 1\nx is :\n");
    for (int i = 0; i < num_of_poly_basis - 1; i++)
        printf("%f ", x_for_Newton_n[i]);
    printf("%f\n", x_for_Newton_n[num_of_poly_basis - 1]);

    printf("Eq is y = %f", x[0]);
    for (int i = 1; i < num_of_poly_basis; i++)
    {
        printf(" + %f*x^%d", x_for_Newton_n[i], i);
    }

    for (int i = 0; i < num_of_input_data; i++)
    {
        y[i] = 0;
    }
    Er = 0;
    for (int i = 0; i < num_of_input_data; i++)
    {
        for (int j = 0; j < num_of_poly_basis; j++)
        {
            y[i] += A[i][j] * x_for_Newton_n[j];
        }
        Er += pow(abs(y[i] - t[i]), 2);
    }
    printf("\nLeast Square Error is %f\nRMSE is %f\n", Er, sqrt(Er / num_of_input_data));
    printf("iter is %d\n", iter);



    /* Version 2 Newton's method - with lambda */
    for (int i = 0; i < num_of_poly_basis; i++)
    {
        x_for_Newton_n[i] = 0;
    }

    iter = -1;
    do
    {
        iter++;

        for (int i = 0; i < num_of_poly_basis; i++)
        {
            tmp_x[i] = 0;
            Udx[i] = 0;
            dx[i] = 0;
        }

        /* ATA_for_Newton x_n - 1 */
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            for (int j = 0; j < num_of_poly_basis; j++)
            {
                tmp_x[i] += ATA/*_for_Newton_orig*/[i][j] * x_for_Newton_n[j];
            }
            tmp_x[i] -= ATb[i];
        }

        /* Use L Udx = - known */
        for (int i = 0; i < num_of_poly_basis; i++)
        {
            Udx[i] -= tmp_x[i];
            for (int k = 0; k < i; k++)
            {
                Udx[i] -= L[i][k] * Udx[k];
            }
        }
        /* U dx = Udx */
        for (int i = num_of_poly_basis - 1; i >= 0; i--)
        {
            dx[i] = Udx[i];
            for (int j = num_of_poly_basis - 1; j > i; j--)
            {
                dx[i] -= dx[j] * U[i][j];
            }
            dx[i] /= U[i][i];
        }

    } while (dx_larger_than_threshold(dx, x_for_Newton_n, num_of_poly_basis));

    printf("\nNewton's method version 2\nx is :\n");
    for (int i = 0; i < num_of_poly_basis - 1; i++)
        printf("%f ", x_for_Newton_n[i]);
    printf("%f\n", x_for_Newton_n[num_of_poly_basis - 1]);

    printf("Eq is y = %f", x[0]);
    for (int i = 1; i < num_of_poly_basis; i++)
    {
        printf(" + %f*x^%d", x_for_Newton_n[i], i);
    }

    for (int i = 0; i < num_of_input_data; i++)
    {
        y[i] = 0;
    }
    Er = 0;
    for (int i = 0; i < num_of_input_data; i++)
    {
        for (int j = 0; j < num_of_poly_basis; j++)
        {
            y[i] += A[i][j] * x_for_Newton_n[j];
        }
        Er += pow(abs(y[i] - t[i]), 2);
    }
    printf("\nLeast Square Error is %f\nRMSE is %f\n", Er, sqrt(Er / num_of_input_data));
    printf("iter is %d\n", iter);

    fclose(fp);
    return 0;
}