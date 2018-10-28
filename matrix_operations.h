#ifndef MATRIX_OPS
#define MATRIX_OPS

/* abitrary-sized matrix operations */

/* construct a matrix data structure */
typedef struct MATRIX
{
    int dimensions[2]; // dimensions[0] x dimensions[1]
    double **data;
} matrix;

/* create the matrix */
matrix *create_matrix(int d0, int d1);

/* printf */
int print_matrix(matrix *A);

/* assign to specific place of a matrix */
int *assign_matrix(matrix *A, matrix *B, int start, bool column);

/* Transpose */
matrix *tran_matrix(matrix *A);

/* Add */
matrix *add_matrix(matrix *A, matrix *B);

/* Matrix multiplication */
matrix *multi_matrix(matrix *A, matrix *B);

/* Free the struct's space */
int free_mat(matrix *A);

/* LU decomposition */
int LU_decomposition(matrix *A, matrix *L, matrix *U);

/* apply LUx = y */
matrix *LUx_y(matrix *A, matrix *y, matrix *L, matrix *U);

/* inverse of an  matrix */
matrix *inverse(matrix *A);

#endif