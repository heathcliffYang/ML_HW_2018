#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include "gaussian_random_number_generator.h"
#include "matrix_operations.h"
#include "mnist_handlers.h"

bool is_converge(double w0, double w1)
{
    static double last_w0 = 0, last_w1 = 0;

    if (fabs(w0 - last_w0) < 0.00001 && fabs(w1 - last_w1) < 0.00001 && w0 != 0 && w1 != 0)
    {
        printf("W Converge!\n");
        return false;
    }

    last_w0 = w0;
    last_w1 = w1;
    return true;
};

bool converge(matrix *p)
{
    static double last_p[10][748] = {};
    double eps = 0;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 748; j++)
        {
            //printf("%f - %f\n", p->data[i][j], last_p[i][j]);
            eps += fabs(p->data[i][j] - last_p[i][j]);
        }
    }
    printf("eps: %f\n", eps);
    if (eps < 0.0001)
    {
        printf("Converge!\n");
        return false;
    }
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 748; j++)
        {
            last_p[i][j] = p->data[i][j];
        }
    }
    return true;
};

bool w_converge(matrix *w)
{
    static double last_w[10][60000] = {};
    double eps = 0;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 60000; j++)
        {
            eps += fabs(w->data[i][j] - last_w[i][j]);
        }
    }
    printf("eps: %f\n", eps);
    if (eps < 0.000000001)
    {
        printf("Converge!\n");
        return false;
    }
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 60000; j++)
        {
            last_w[i][j] = w->data[i][j];
        }
    }
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
        int num_of_I_error = 0, num_of_II_error = 0, i1 = 0, i2 = 0, iter = 0, num_of_change_cluster = 0, converge_alarm = 0;
        double learning_rate = 0.01, Sensitivity = 0, Specificity = 0;
        matrix *ans_sheet = create_matrix(num_of_data, 2);
        matrix *last_ans_sheet = create_matrix(num_of_data, 2);

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
                ans_sheet->data[i][0] = 1;
                ans_sheet->data[i][1] = 2;
                if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                {
                    I_error->data[i1][0] = D1->data[i][0];
                    I_error->data[i1][1] = D1->data[i][1];
                    i1++;
                    ans_sheet->data[i][0] = 2;
                }
                if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                {
                    II_error->data[i2][0] = D2->data[i][0];
                    II_error->data[i2][1] = D2->data[i][1];
                    i2++;
                    ans_sheet->data[i][1] = 1;
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
                exp_part_I->data[i][0] = -learning_rate * ((1 / (1 + exp(-I_error_W->data[i][0]))) - 0);
            }
            //printf("exp I\n");
            //print_matrix(exp_part_I);
            matrix *gradient_f_I = multi_matrix(I_error_T, exp_part_I);
            add_assign_matrix(gradient_f_I, w, 0, true);
            // printf("gradient_f_I\n");
            // print_matrix(gradient_f_I);

            for (int i = 0; i < num_of_II_error; i++)
            {
                exp_part_II->data[i][0] = -learning_rate * ((1 / (1 + exp(-II_error_W->data[i][0]))) - 1);
            }
            // printf("exp II\n");
            // print_matrix(exp_part_II);
            matrix *gradient_f_II = multi_matrix(II_error_T, exp_part_II);
            add_assign_matrix(gradient_f_II, w, 0, true);
            // printf("gradient_f_II\n");
            // print_matrix(gradient_f_II);
            // printf("w\n");
            // print_matrix(w);
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

            /* check if converge by classification becomes steady */
            if (iter == 1)
            {
                assign_matrix(ans_sheet, last_ans_sheet, 0, true);
                continue;
            }
            num_of_change_cluster = 0;
            for (int i = 0; i < num_of_data; i++)
            {
                if (ans_sheet->data[i][0] != last_ans_sheet->data[i][0])
                {
                    num_of_change_cluster++;
                }
                if (ans_sheet->data[i][1] != last_ans_sheet->data[i][1])
                {
                    num_of_change_cluster++;
                }
            }
            if ((double)(num_of_change_cluster / num_of_data) < 0.02)
            {
                converge_alarm++;
            }
            else
            {
                converge_alarm = 0;
            }
            if (converge_alarm > 15 && (double)(num_of_change_cluster / num_of_data) < 0.02)
            {
                printf("The ratio of data changing cluster is less than %f, Converge!\n", (double)(num_of_change_cluster / num_of_data));
                break;
            }
            assign_matrix(ans_sheet, last_ans_sheet, 0, true);

        } //while (iter < 10);
        while (is_converge(w->data[0][0], w->data[1][0]));

        /* Newton's method */
        printf("\n\nContinue to do Newton's method?[y/n]\n");
        char ans[2];
        if (scanf("%s", ans) && ans[0] == 'y')
        {

            /* reset parameters*/
            w->data[0][0] = 0;
            w->data[1][0] = 0;
            num_of_I_error = 0, num_of_II_error = 0, iter = 0;
            Sensitivity = 0, Specificity = 0;
            int num_of_error = 0;
            int error_type_I_index[1000] = {};
            num_of_change_cluster = 0;
            converge_alarm = 0;
            learning_rate = 0.01;

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

                /* TODO: append these errors into a matrix */

                for (int i = 0; i < 1000; i++)
                {
                    error_type_I_index[i] = 0;
                }
                num_of_I_error = 0, num_of_II_error = 0, i1 = 0, i2 = 0;
                for (int i = 0; i < num_of_data; i++)
                {
                    ans_sheet->data[i][0] = 1;
                    ans_sheet->data[i][1] = 2;
                    if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                    {
                        num_of_I_error++;
                        ans_sheet->data[i][0] = 2;
                    }
                    if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                    {
                        num_of_II_error++;
                        ans_sheet->data[i][1] = 1;
                    }
                }
                num_of_error = num_of_I_error + num_of_II_error;
                printf("Type I error: %d\nType II error: %d\n", num_of_I_error, num_of_II_error);

                Sensitivity = (double)(num_of_data - num_of_II_error) / (double)num_of_data,
                Specificity = (double)(num_of_data - num_of_I_error) / (double)num_of_data;

                // printf("Confusion matrix:\n%d | %d\n%d | %d\nSensitivity: %f\nSpecificity: %f\n\n",
                //        num_of_data - num_of_II_error, num_of_II_error, num_of_I_error, num_of_data - num_of_I_error,
                //        Sensitivity, Specificity);
                if (num_of_error == 0)
                    break;

                matrix *error = create_matrix(num_of_error, 2);
                for (int i = 0; i < num_of_data; i++)
                {
                    if (1 / (1 + exp(-classify_result1->data[i][0])) >= 0.5)
                    {
                        //I_error->data[i1][0] = D1->data[i][0];
                        //I_error->data[i1][1] = D1->data[i][1];
                        error->data[i1][0] = D1->data[i][0];
                        error->data[i1][1] = D1->data[i][1];
                        error_type_I_index[i1] = 1;
                        i1++;
                    }
                    if (1 / (1 + exp(-classify_result2->data[i][0])) < 0.5)
                    {
                        //II_error->data[i2][0] = D2->data[i][0];
                        //II_error->data[i2][1] = D2->data[i][1];
                        error->data[i1][0] = D2->data[i][0];
                        error->data[i1][1] = D2->data[i][1];
                        i1++;
                        //i2++;
                    }
                }
                // printf("\nerror\n");
                // print_matrix(error);

                matrix *error_W = multi_matrix(error, w);
                //printf("error_W\n");
                //print_matrix(error_W);
                matrix *error_T = tran_matrix(error);
                matrix *exp_part = create_matrix(num_of_error, 1);
                matrix *Hessian_D = create_matrix(num_of_error, num_of_error);
                for (int i = 0; i < num_of_error; i++)
                {
                    if (error_type_I_index[i] == 1)
                    {
                        exp_part->data[i][0] = learning_rate * (-((1 / (1 + exp(-error_W->data[i][0]))) - 0));
                    }
                    else
                    {
                        exp_part->data[i][0] = learning_rate * (-((1 / (1 + exp(-error_W->data[i][0]))) - 1));
                    }
                    /* Hessian's D */
                    Hessian_D->data[i][i] = exp(-error_W->data[i][0]) / pow(1 + exp(-error_W->data[i][0]), 2);
                    // printf("exp(-error_W): %f\n", exp(-error_W->data[i][0]));
                }
                // printf("Hessian_D\n");
                // print_matrix(Hessian_D);
                // printf("exp_part\n");
                // print_matrix(exp_part);
                matrix *gradient_f = multi_matrix(error_T, exp_part);
                // printf("gradient_f\n");
                // print_matrix(gradient_f);
                matrix *Phi_D = multi_matrix(error_T, Hessian_D);
                // printf("Phi_D\n");
                // print_matrix(Phi_D);
                matrix *Hessian = multi_matrix(Phi_D, error);
                // printf("Hessian\n");
                // print_matrix(Hessian);
                matrix *H_inverse = inverse(Hessian);

                if (H_inverse == NULL)
                {
                    printf("Cannot find the Hessian inverse\n");
                    return -1;
                }
                matrix *H_F = multi_matrix(H_inverse, gradient_f);
                // printf("H_F\n");
                // print_matrix(H_F);
                // for (int z = 0; z < 2; z++)
                // {
                //     for (int r = 0; r < 2; r++)
                //     {
                //         H_F->data[z][r] = H_F->data[z][r] * lr;
                //     }
                // }
                // printf("H_F\n");
                // print_matrix(H_F);
                add_assign_matrix(H_F, w, 0, true);

                /* check if converge by classification becomes steady */
                if (iter == 1)
                {
                    assign_matrix(ans_sheet, last_ans_sheet, 0, true);
                    continue;
                }
                num_of_change_cluster = 0;
                for (int i = 0; i < num_of_data; i++)
                {
                    if (ans_sheet->data[i][0] != last_ans_sheet->data[i][0])
                    {
                        num_of_change_cluster++;
                    }
                    if (ans_sheet->data[i][1] != last_ans_sheet->data[i][1])
                    {
                        num_of_change_cluster++;
                    }
                }
                if ((double)(num_of_change_cluster / num_of_data) < 0.10)
                {
                    converge_alarm++;
                    printf("Converge alarm!\n");
                }
                else
                {
                    converge_alarm = 0;
                    printf("Reset converge alarm!\n");
                }
                if (converge_alarm > 10 && (double)(num_of_change_cluster / num_of_data) < 0.10)
                {
                    printf("The ratio of data changing cluster is less than %f, Converge!\n", (double)(num_of_change_cluster / num_of_data));
                    break;
                }
                assign_matrix(ans_sheet, last_ans_sheet, 0, true);

                free(Hessian);
                free(H_inverse);
                free(H_F);
                free(Phi_D);
                free(Hessian_D);
                free(error_T);
                free(error_W);
                free(exp_part);
                free(classify_result1);
                //free(I_error);
                free(classify_result2);
                //free(II_error);
                free(error);
            } while (true);
            //while (is_converge(w->data[0][0], w->data[1][0]));
        }

        free(D1);
        free(D2);
        free(w);
        free(ans_sheet);
        free(last_ans_sheet);
    }
    /* mode 2 - EM algorithm */
    else if (atoi(argv[1]) == 2)
    {
        uint8_t *train_labels = NULL, **train_images = NULL, middle_pixel_value = 70;

        int read_success = 0, iter = 0;
        unsigned int train_num_of_data = 0, train_num_of_pixels = 0;

        double wki = 0, threshold = atof(argv[2]);
        printf("Threshold: %f\n", threshold);

        read_success = data_read("HW2_input/train-labels.idx1-ubyte", "HW2_input/train-images.idx3-ubyte",
                                 &train_labels, &train_images, &train_num_of_data, &train_num_of_pixels);

        matrix *p = create_matrix(10, train_num_of_pixels);
        matrix *w = create_matrix(10, train_num_of_data);
        matrix *lambda = create_matrix(10, 1);
        double weighted_likelihood[10] = {}, sum_likelihood = 0;

        /* Check my input?? */
        if (read_success == -1)
        {
            printf("Fail to read files\n");
        }

        printf("Guess parameters start\n");
        /* Guess parameters */
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < train_num_of_pixels; j++)
            {
                p->data[i][j] = Marsaglia_gaussian_rand_generator(0.5, 0.2);
            }
            lambda->data[i][0] = 0.1;
        }

        printf("Guess parameters finished\n");
        do
        {
            iter++;
            /* E - step: calculate responsibility */
            printf("iter %d - E-step -> ", iter);
            for (int k = 0; k < train_num_of_data; k++)
            {
                sum_likelihood = 0;
                for (int i = 0; i < 10; i++)
                {
                    weighted_likelihood[i] = log(lambda->data[i][0]);
                    for (int j = 0; j < train_num_of_pixels; j++)
                    {
                        if (train_images[k][j] > middle_pixel_value)
                        {
                            if ((p->data[i][j]) > threshold)
                                weighted_likelihood[i] += log(p->data[i][j]);
                            else
                                weighted_likelihood[i] += log(threshold);
                        }
                        else
                        {
                            if ((1 - p->data[i][j]) > threshold)
                                weighted_likelihood[i] += log(1 - p->data[i][j]);
                            else
                                weighted_likelihood[i] += log(threshold);
                        }
                    }
                    sum_likelihood += weighted_likelihood[i];
                }
                //printf("\nw of %d: ", k);
                for (int i = 0; i < 10; i++)
                {
                    w->data[i][k] = weighted_likelihood[i] / sum_likelihood;
                    //printf("%.2f, %.2f| ", w->data[i][k], weighted_likelihood[i]);
                }
                //printf("\n");
            }

            /* M - step */
            /* reset p */
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < train_num_of_pixels; j++)
                {
                    p->data[i][j] = 0;
                }
            }
            printf("M-step\n");
            for (int i = 0; i < 10; i++)
            {
                wki = 0;
                for (int k = 0; k < train_num_of_data; k++)
                {
                    wki += w->data[i][k];
                    for (int j = 0; j < train_num_of_pixels; j++)
                    {
                        p->data[i][j] += (train_images[k][j] > middle_pixel_value) * w->data[i][k];
                    }
                }

                for (int j = 0; j < train_num_of_pixels; j++)
                {
                    p->data[i][j] /= wki;
                }

                lambda->data[i][0] = wki / train_num_of_data;
                printf("%f ", lambda->data[i][0]);
            }
            printf("\n");
        } while (w_converge(w));
        //while (iter < 3);

        int ans[60000] = {};
        double max = 0;
        int confusion[10][4] = {};
        for (int k = 0; k < train_num_of_data; k++)
        {
            max = 0;
            for (int i = 0; i < 10; i++)
            {
                if (w->data[i][k] > max)
                {
                    max = w->data[i][k];
                    ans[k] = i;
                }
            }
            // if (k < 12000 && k > 10000)
            //     printf("ans[%d]: %d - truth: %d\n", k, ans[k], train_labels[k]);
            for (int i = 0; i < 10; i++)
            {
                /* positive at class-i */
                if (ans[k] == i)
                {
                    /* TP at class-i */
                    if (i == train_labels[k])
                    {
                        confusion[i][0]++;
                    }
                    else /* FP at class-i, FN at class-train_labels[k] */
                    {
                        confusion[i][2]++;
                        confusion[train_labels[k]][1]++;
                    }
                }
                else
                {
                    if (i != train_labels[k])
                        confusion[i][3]++;
                }
            }
        }

        for (int i = 0; i < 10; i++)
        {
            printf("Class %d\nConfusion matrix:\n%d | %d\n%d | %d\nSensitivity: %f\nSpecificity: %f\n\n",
                   i,
                   confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
                   (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
                   (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
        }

        free(p);
        free(w);
        free(lambda);
        free(train_labels);
        free(train_images);
    }

    return 0;
}