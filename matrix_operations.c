#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "matrix_operations.h"

/* create the matrix */
matrix *create_matrix(int d0, int d1)
{

    int i, j;
    matrix *A = malloc(sizeof(matrix));

    /* define the result matrix's dimensions */
    A->dimensions[0] = d0;
    A->dimensions[1] = d1;

    /* allocate memory to this result */
    //printf("Allocate memory, %d * %d\n", d0, d1);
    A->data = (double **)malloc(sizeof(double *) * A->dimensions[0] + A->dimensions[0] * A->dimensions[1] * sizeof(double));
    double *pData;
    //printf("Rearrange pointer\n");
    for (i = 0, pData = (double *)((A->data) + d0); i < A->dimensions[0]; i++, pData += d1)
    {
        (A->data)[i] = pData;
    }
    /* initialize A */
    for (i = 0; i < A->dimensions[0]; i++)
    {
        for (j = 0; j < A->dimensions[1]; j++)
        {
            A->data[i][j] = 0;
        }
    }
    //printf("end\n");
    return A;
};

/* printf */
int print_matrix(matrix *A)
{
    int i = 0, j = 0;
    for (i = 0; i < A->dimensions[0]; i++)
    {
        for (j = 0; j < A->dimensions[1]; j++)
        {
            printf("%f ", A->data[i][j]);
        }
        printf("\n");
    }
    return 0;
};

/* assign to specific place of a matrix */
/* B <- A */
int *assign_matrix(matrix *A, matrix *B, int start, bool column)
{
    if (column == true)
    {
        //printf("assign\nA\n");
        //print_matrix(A);
        //printf("B\n");
        //print_matrix(B);
        for (int i = start, k = 0; k < A->dimensions[1]; i++, k++)
        {
            for (int j = 0; j < A->dimensions[0]; j++)
            {
                B->data[j][i] = A->data[j][k];
            }
        }
    }
    return 0;
};

/* assign to specific place of a matrix */
/* B <- A */
int *add_assign_matrix(matrix *A, matrix *B, int start, bool column)
{
    if (column == true)
    {
        //printf("assign\nA\n");
        //print_matrix(A);
        //printf("B\n");
        //print_matrix(B);
        for (int i = start, k = 0; k < A->dimensions[1]; i++, k++)
        {
            for (int j = 0; j < A->dimensions[0]; j++)
            {
                B->data[j][i] += A->data[j][k];
            }
        }
    }
    return 0;
};

/* Transpose */
matrix *tran_matrix(matrix *A)
{
    matrix *B = create_matrix(A->dimensions[1], A->dimensions[0]);
    for (int i = 0; i < A->dimensions[0]; i++)
    {
        for (int j = 0; j < A->dimensions[1]; j++)
        {
            B->data[j][i] = A->data[i][j];
        }
    }
    return B;
};

/* Add */
matrix *add_matrix(matrix *A, matrix *B)
{
    matrix *result = create_matrix(A->dimensions[0], A->dimensions[1]);
    for (int i = 0; i < A->dimensions[0]; i++)
    {
        for (int j = 0; j < A->dimensions[1]; j++)
        {
            result->data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
    return result;
};

/* Matrix multiplication */
/* A x B */
matrix *multi_matrix(matrix *A, matrix *B)
{
    matrix *result = create_matrix(A->dimensions[0], B->dimensions[1]);
    int i = 0, j = 0, k = 0;

    for (i = 0; i < A->dimensions[0]; i++)
    {
        for (j = 0; j < B->dimensions[1]; j++)
        {
            for (k = 0; k < B->dimensions[0]; k++)
            {
                result->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return result;
};

/* Free the struct's space */
int free_mat(matrix *A)
{
    free(A->data);
    return 0;
};

/* LU decomposition */
/* handle the '0' case */
int LU_decomposition(matrix *A, matrix *L, matrix *U)
{
    for (int k = 0; k < A->dimensions[0]; k++)
    {
        for (int j = k; j < A->dimensions[0]; j++)
        {
            U->data[k][j] = A->data[k][j];
            for (int s = 0; s <= k - 1; s++)
                U->data[k][j] -= L->data[k][s] * U->data[s][j];
        }

        L->data[k][k] = 1;
        for (int i = k + 1; i < A->dimensions[0]; i++)
        {
            L->data[i][k] = A->data[i][k];
            for (int s = 0; s <= k - 1; s++)
                L->data[i][k] -= L->data[i][s] * U->data[s][k];
            if (U->data[k][k] == 0)
            {
                printf("RRRRRRR\nRRRRRRR\nRRRRRRR\nRRRRRRR\nRRRRRRR\n");
                return -1;
            }

            L->data[i][k] /= U->data[k][k];
        }
    }
    return 0;
};

/* TODO apply LUx = y */
matrix *LUx_y(matrix *A, matrix *y, matrix *L, matrix *U)
{
    /* Lx' = y */
    matrix *x_p = create_matrix(y->dimensions[0], y->dimensions[1]);
    for (int i = 0; i < L->dimensions[0]; i++)
    {
        /* TODO overload vector?? */
        x_p->data[i][0] = y->data[i][0];
        for (int k = 0; k < i; k++)
        {
            x_p->data[i][0] -= L->data[i][k] * x_p->data[k][0];
        }
    }
    /* Ux = x_p */
    matrix *x = create_matrix(y->dimensions[0], y->dimensions[1]);
    for (int i = U->dimensions[0] - 1; i >= 0; i--)
    {
        x->data[i][0] = x_p->data[i][0];
        for (int j = U->dimensions[1] - 1; j > i; j--)
        {
            x->data[i][0] -= x->data[j][0] * U->data[i][j];
        }
        x->data[i][0] /= U->data[i][i];
    }
    free_mat(x_p);
    return x;
};

/* inverse of an  matrix */
matrix *inverse(matrix *A)
{
    matrix *L = create_matrix(A->dimensions[0], A->dimensions[1]);
    matrix *U = create_matrix(A->dimensions[0], A->dimensions[1]);
    matrix *y = create_matrix(A->dimensions[0], 1);
    matrix *x;
    matrix *inverse = create_matrix(A->dimensions[0], A->dimensions[1]);

    if (LU_decomposition(A, L, U) == -1)
    {
        return NULL;
    }

    //printf("L\n");
    //print_matrix(L);
    //printf("U\n");
    //print_matrix(U);

    for (int i = 0; i < A->dimensions[0]; i++)
    {
        /* LUx = y, y = I[i][j] j from 0 to N */
        if (i == 0)
        {
            y->data[0][0] = 1;
        }
        else
        {
            y->data[i - 1][0] = 0;
            y->data[i][0] = 1;
        }

        x = LUx_y(A, y, L, U);
        //printf("x\n");
        //print_matrix(x);
        assign_matrix(x, inverse, i, true);
        free_mat(x);
    }
    //printf("inverse\n");
    //print_matrix(inverse);
    free_mat(L);
    free_mat(U);
    free_mat(y);
    return inverse;
};