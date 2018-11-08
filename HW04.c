#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "gaussian_random_number_generator.h"
#include "matrix_operations.h"

bool is_converge(double Sensitivity, double Specificity)
{
    static double last_sen = 0, last_spe = 0;

    if (Sensitivity >= 0.99 && Specificity >= 0.99)
    {
        printf("Converge!\n");
        return false;
    }
    else if (fabs(Sensitivity - last_sen) < 0.05 && fabs(Specificity - last_spe) < 0.05 && Sensitivity > 0.7 && Specificity > 0.7)
    {
        printf("Converge!\n");
        return false;
    }

    last_sen = Sensitivity;
    last_spe = Specificity;
    return true;
};

int main(int argc, char **argv)
{
    srand(time(NULL));
    /* mode 1 - Steepest gradient descent */
    if (atoi(argv[1]) == 1)
    {
        int num_of_data = atoi(argv[2]);
        printf("There are 2 set of %d data\n\n", num_of_data);

        /* argv [3 ~ 10]  */
        /* [mean][var] */
        double mean_var[4][2] = {};
        for (int i = 3; i <= 10; i++)
        {
            mean_var[(i - 3) / 2][1 * (i % 2 == 0)] = atof(argv[i]);
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                printf("%f ", mean_var[i][j]);
            }
            printf("\n");
        }

        /* generate data */
        matrix *D1 = create_matrix(num_of_data, 2);
        matrix *D2 = create_matrix(num_of_data, 2);

        for (int i = 0; i < num_of_data; i++)
        {
            D1->data[i][0] = Marsaglia_gaussian_rand_generator(mean_var[0][0], mean_var[0][1]);
            D1->data[i][1] = Marsaglia_gaussian_rand_generator(mean_var[1][0], mean_var[1][1]);
            D2->data[i][0] = Marsaglia_gaussian_rand_generator(mean_var[2][0], mean_var[2][1]);
            D2->data[i][1] = Marsaglia_gaussian_rand_generator(mean_var[3][0], mean_var[3][1]);
        }

        /*
        printf("D1\n");
        print_matrix(D1);
        printf("D2\n");
        print_matrix(D2);
        printf("\n");
        */

        /* initial w */
        matrix *w = create_matrix(2, 1);
        int num_of_I_error = 0, num_of_II_error = 0, i1 = 0, i2 = 0, iter = 0;
        double learning_rate = 1, Sensitivity = 0, Specificity = 0;

        /* Steepest gradient descent */
        do
        {
            iter++;
            printf("iter %d\n", iter);
            printf("weights:\n");
            print_matrix(w);
            /* -- Collect type I and type II error */
            printf("Collect misclassified data...\n");
            matrix *classify_result1 = multi_matrix(D1, w); //  if D1 == 1, type I error  (correct is 0)
            matrix *classify_result2 = multi_matrix(D2, w); //  if D2 == 0, type II error (correct is 1)
            //printf("Classify result1\n");
            //print_matrix(classify_result1);
            //printf("Classify result2\n");
            //print_matrix(classify_result2);
            /* TODO: append these errors into a matrix */
            num_of_I_error = 0, num_of_II_error = 0, i1 = 0, i2 = 0;
            for (int i = 0; i < num_of_data; i++)
            {
                if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                    num_of_I_error++;
                if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                    num_of_II_error++;
            }
            printf("Type I error: %d\nType II error: %d\n", num_of_I_error, num_of_II_error);

            Sensitivity = (double)(num_of_data - num_of_II_error) / (double)num_of_data,
            Specificity = (double)(num_of_data - num_of_I_error) / (double)num_of_data;
            printf("Confusion matrix:\n%d | %d\n%d | %d\nSensitivity: %f\nSpecificity: %f\n\n",
                   num_of_data - num_of_II_error, num_of_II_error, num_of_I_error, num_of_data - num_of_I_error,
                   Sensitivity, Specificity);

            matrix *I_error = create_matrix(num_of_I_error, 2);
            matrix *II_error = create_matrix(num_of_II_error, 2);
            for (int i = 0; i < num_of_data; i++)
            {
                if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                {
                    I_error->data[i1][0] = D1->data[i][0];
                    I_error->data[i1][1] = D1->data[i][1];
                    i1++;
                }
                if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                {
                    II_error->data[i2][0] = D2->data[i][0];
                    II_error->data[i2][1] = D2->data[i][1];
                    i2++;
                }
            }

            /* -- 1/(1+exp(-WTx)) -  */
            matrix *I_error_W = multi_matrix(I_error, w);
            matrix *II_error_W = multi_matrix(II_error, w);
            matrix *I_error_T = tran_matrix(I_error);
            matrix *II_error_T = tran_matrix(II_error);
            matrix *exp_part_I = create_matrix(num_of_I_error, 1);
            matrix *exp_part_II = create_matrix(num_of_II_error, 1);
            //printf("I_error\n");
            //print_matrix(I_error);
            //printf("II_error\n");
            //print_matrix(II_error);
            for (int i = 0; i < num_of_I_error; i++)
            {
                exp_part_I->data[i][0] = learning_rate * (-((1 / (1 + exp(-I_error_W->data[i][0]))) - 0));
            }
            //printf("exp I\n");
            //print_matrix(exp_part_I);
            matrix *gradient_f_I = multi_matrix(I_error_T, exp_part_I);
            add_assign_matrix(gradient_f_I, w, 0, true);
            //printf("gradient_f_I\n");
            //print_matrix(gradient_f_I);

            for (int i = 0; i < num_of_II_error; i++)
            {
                exp_part_II->data[i][0] = learning_rate * (-((1 / (1 + exp(-II_error_W->data[i][0]))) - 1));
            }
            //printf("exp II\n");
            //print_matrix(exp_part_II);
            matrix *gradient_f_II = multi_matrix(II_error_T, exp_part_II);
            add_assign_matrix(gradient_f_II, w, 0, true);
            //printf("gradient_f_II\n");
            //print_matrix(gradient_f_II);
            //printf("w\n");
            //print_matrix(w);

            free(classify_result1);
            free(I_error);
            free(I_error_T);
            free(I_error_W);
            free(exp_part_I);

            free(classify_result2);
            free(II_error);
            free(II_error_T);
            free(II_error_W);
            free(exp_part_II);
        } //while (iter < 10);
        while (is_converge(Sensitivity, Specificity));

        /* Newton's method */
        printf("\n\nContinue to do Newton's method?[Y/N]\n");
        char ans[2];
        if (scanf("%s", ans) && ans[0] == 'Y')
        {

            /* reset parameters*/
            w->data[0][0] = 0;
            w->data[1][0] = 0;
            num_of_I_error = 0, num_of_II_error = 0, iter = 0;
            Sensitivity = 0, Specificity = 0;
            double tmp = 0;
            do
            {
                iter++;
                printf("\niter %d\n", iter);
                printf("weights:\n");
                print_matrix(w);
                /* -- Collect type I and type II error */
                //printf("Collect misclassified data...\n");
                matrix *classify_result1 = multi_matrix(D1, w); //  if D1 == 1, type I error  (correct is 0)
                matrix *classify_result2 = multi_matrix(D2, w); //  if D2 == 0, type II error (correct is 1)
                //printf("Classify result1\n");
                //print_matrix(classify_result1);
                //printf("Classify result2\n");
                //print_matrix(classify_result2);
                /* TODO: append these errors into a matrix */
                num_of_I_error = 0, num_of_II_error = 0, i1 = 0, i2 = 0;
                for (int i = 0; i < num_of_data; i++)
                {
                    if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                        num_of_I_error++;
                    if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                        num_of_II_error++;
                }
                printf("Type I error: %d\nType II error: %d\n", num_of_I_error, num_of_II_error);

                Sensitivity = (double)(num_of_data - num_of_II_error) / (double)num_of_data,
                Specificity = (double)(num_of_data - num_of_I_error) / (double)num_of_data;

                printf("Confusion matrix:\n%d | %d\n%d | %d\nSensitivity: %f\nSpecificity: %f\n\n",
                       num_of_data - num_of_II_error, num_of_II_error, num_of_I_error, num_of_data - num_of_I_error,
                       Sensitivity, Specificity);

                matrix *I_error = create_matrix(num_of_I_error, 2);
                matrix *II_error = create_matrix(num_of_II_error, 2);
                for (int i = 0; i < num_of_data; i++)
                {
                    if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                    {
                        I_error->data[i1][0] = D1->data[i][0];
                        I_error->data[i1][1] = D1->data[i][1];
                        i1++;
                    }
                    if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                    {
                        II_error->data[i2][0] = D2->data[i][0];
                        II_error->data[i2][1] = D2->data[i][1];
                        i2++;
                    }
                }

                /* -- 1/(1+exp(-WTx)) -  */
                matrix *I_error_W = multi_matrix(I_error, w);
                matrix *II_error_W = multi_matrix(II_error, w);
                matrix *I_error_T = tran_matrix(I_error);
                matrix *II_error_T = tran_matrix(II_error);
                matrix *exp_part_I = create_matrix(num_of_I_error, 1);
                matrix *exp_part_II = create_matrix(num_of_II_error, 1);
                //printf("I_error\n");
                //print_matrix(I_error);
                //printf("II_error\n");
                //print_matrix(II_error);
                for (int i = 0; i < num_of_I_error; i++)
                {
                    exp_part_I->data[i][0] = learning_rate * (-((1 / (1 + exp(-I_error_W->data[i][0]))) - 0));
                }
                //printf("exp I\n");
                //print_matrix(exp_part_I);
                matrix *gradient_f_I = multi_matrix(I_error_T, exp_part_I);
                //add_assign_matrix(gradient_f_I, w, 0, true);
                //printf("gradient_f_I\n");
                //print_matrix(gradient_f_I);
                if (num_of_I_error != 0)
                {
                    /* Hessian */
                    matrix *Hessian_I = create_matrix(2, 2);
                    tmp = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        for (int k = 0; k < 2; k++)
                        {
                            for (int i = 0; i < num_of_I_error; i++)
                            {

                                tmp = exp(-w->data[k][0] * I_error->data[i][k]);
                                printf("W: %f, Phi: %f, -WPhi: %f, tmp: %f\n", w->data[k][0], I_error->data[i][k], -w->data[k][0] * I_error->data[i][k], tmp);
                                Hessian_I->data[j][k] += I_error->data[i][j] * I_error->data[i][k] * (tmp / pow((1 + tmp), 2));
                            }
                        }
                    }
                    print_matrix(Hessian_I);
                    matrix *H_I_inverse = inverse(Hessian_I);

                    if (H_I_inverse == NULL)
                    {
                        printf("Cannot find the Hessian I inverse\n");
                        break;
                    }
                    printf("HIinverse\n");
                    print_matrix(H_I_inverse);
                    matrix *H_F_I = multi_matrix(H_I_inverse, gradient_f_I);
                    add_assign_matrix(H_F_I, w, 0, true);
                    printf("H_F_I\n");
                    print_matrix(H_F_I);
                    free(Hessian_I);
                    free(H_I_inverse);
                    free(H_F_I);
                }

                for (int i = 0; i < num_of_II_error; i++)
                {
                    exp_part_II->data[i][0] = learning_rate * (-((1 / (1 + exp(-II_error_W->data[i][0]))) - 1));
                }
                //printf("exp II\n");
                //print_matrix(exp_part_II);
                matrix *gradient_f_II = multi_matrix(II_error_T, exp_part_II);
                //add_assign_matrix(gradient_f_II, w, 0, true);
                //printf("gradient_f_II\n");
                //print_matrix(gradient_f_II);
                //printf("w\n");
                //print_matrix(w);

                if (num_of_II_error != 0)
                {
                    /* Hessian */
                    matrix *Hessian_II = create_matrix(2, 2);
                    tmp = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        for (int k = 0; k < 2; k++)
                        {
                            for (int i = 0; i < num_of_II_error; i++)
                            {
                                tmp = exp(-w->data[k][0] * II_error->data[i][k]);
                                Hessian_II->data[j][k] += II_error->data[i][j] * II_error->data[i][k] * (tmp / pow((1 + tmp), 2));
                            }
                        }
                    }
                    matrix *H_II_inverse = inverse(Hessian_II);
                    if (H_II_inverse == NULL)
                    {
                        printf("Cannot find the Hessian II inverse\n");
                        break;
                    }
                    matrix *H_F_II = multi_matrix(H_II_inverse, gradient_f_II);
                    printf("H_F_II\n");
                    print_matrix(H_F_II);
                    add_assign_matrix(H_F_II, w, 0, true);

                    free(Hessian_II);
                    free(H_II_inverse);
                    free(H_F_II);
                }

                free(classify_result1);
                free(I_error);
                free(I_error_T);
                free(I_error_W);
                free(exp_part_I);
                free(classify_result2);
                free(II_error);
                free(II_error_T);
                free(II_error_W);
                free(exp_part_II);
            } //while (iter < 10);
            while (is_converge(Sensitivity, Specificity));
        }

        free(D1);
        free(D2);
        free(w);
    }
    /* mode 2 */
    else if (atoi(argv[1]) == 2)
    {
        int num_of_data = atoi(argv[2]);
        printf("There are 2 set of %d data\n\n", num_of_data);

        /* argv [3 ~ 10]  */
        /* [mean][var] */
        double mean_var[4][2] = {};
        for (int i = 3; i <= 10; i++)
        {
            mean_var[(i - 3) / 2][1 * (i % 2 == 0)] = atof(argv[i]);
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                printf("%f ", mean_var[i][j]);
            }
            printf("\n");
        }

        /* generate data */
        matrix *D1 = create_matrix(num_of_data, 2);
        matrix *D2 = create_matrix(num_of_data, 2);

        for (int i = 0; i < num_of_data; i++)
        {
            D1->data[i][0] = Marsaglia_gaussian_rand_generator(mean_var[0][0], mean_var[0][1]);
            D1->data[i][1] = Marsaglia_gaussian_rand_generator(mean_var[1][0], mean_var[1][1]);
            D2->data[i][0] = Marsaglia_gaussian_rand_generator(mean_var[2][0], mean_var[2][1]);
            D2->data[i][1] = Marsaglia_gaussian_rand_generator(mean_var[3][0], mean_var[3][1]);
        }

        /*
        printf("D1\n");
        print_matrix(D1);
        printf("D2\n");
        print_matrix(D2);
        printf("\n");
        */

        free(D1);
        free(D2);
    }

    return 0;
}