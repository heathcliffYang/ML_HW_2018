#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include "svm.h"
#include <unistd.h>

/* input file */
char target_input_train[] = "T_train.csv";
char X_input_train[] = "X_train.csv";
char target_input_test[] = "T_test.csv";
char X_input_test[] = "X_test.csv";

int construct_svm_problem(char *target_input, char *X_input, struct svm_problem *svm_problem)
{
    char buf[10], linebuf[4000];
    int num_of_data = 0;
    int attribute_line = 784 + 1;

    FILE *fp_target = fopen(target_input, "r");
    FILE *fp_X = fopen(X_input, "r");

    if (fp_target == NULL)
    {
        printf("Fail to open the target-input file\n");
        return -1;
    }
    if (!fp_X)
    {
        printf("Fail to open the X-input file\n");
        return -1;
    }

    /* Run out to check the total # of data point (an image) */
    while (fgets(buf, sizeof(buf), fp_target) != NULL)
    {
        buf[sizeof(buf) - 1] = '\0';
        num_of_data++;
    }
    fseek(fp_target, 0, SEEK_SET);

    printf("There are %d data points\n", num_of_data);
    double *y = (double *)malloc(sizeof(double) * num_of_data);
    struct svm_node **X = (struct svm_node **)malloc(sizeof(struct svm_node *) * num_of_data + sizeof(struct svm_node) * num_of_data * attribute_line);
    if (!X || !y)
    {
        printf("fail to malloc\n");
    }
    struct svm_node *pStruct;
    int j = 0;
    for (j = 0, pStruct = (struct svm_node *)(X + num_of_data); j < num_of_data; j++, pStruct += attribute_line)
    {
        X[j] = pStruct;
    }
    char *ptr;
    for (int i = 0; i < num_of_data; i++)
    {
        printf("Do data %d\n", i);
        fgets(buf, sizeof(buf), fp_target);
        ptr = strtok(buf, "\n");
        y[i] = atof(ptr);
        fgets(linebuf, sizeof(linebuf), fp_X);
        ptr = strtok(linebuf, "\n");
        for (j = 0, ptr = strtok(ptr, ","); j < attribute_line - 1; j++, ptr = strtok(NULL, ","))
        {

            X[i][j].index = j;
            X[i][j].value = atof(ptr);
        }
        X[i][j].index = -1;
        X[i][j].value = 0;
        //printf("\n");
    }
    svm_problem->x = X;
    svm_problem->y = y;
    svm_problem->l = num_of_data;

    printf("Finish construction\n");
    fclose(fp_target);
    fclose(fp_X);
    return 0;
};

void do_cross_validation(struct svm_problem *prob, struct svm_parameter *param, FILE *fp)
{
    int i;
    int total_correct = 0;
    double *target = (double *)malloc(prob->l * sizeof(double));
    svm_cross_validation(prob, param, 5, target);
    for (i = 0; i < prob->l; i++)
        if (target[i] == prob->y[i])
            ++total_correct;
    printf("    Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob->l);
    free(target);
}

int main(int argc, char const *argv[])
{
    const char *error_msg;
    FILE *log_file = fopen("RBF_log.txt", "a+");
    if (!log_file)
    {
        printf("Fail to create log file\n");
    }

    /* Construct input data structure */
    struct svm_problem train_problem;
    struct svm_problem test_problem;

    /* Read files */
    construct_svm_problem(target_input_train, X_input_train, &train_problem);
    construct_svm_problem(target_input_test, X_input_test, &test_problem);

    /* Classify these images into 5 classes */
    struct svm_parameter parameters;
    parameters.svm_type = C_SVC;
    parameters.shrinking = 1;
    parameters.eps = 0.001;
    parameters.cache_size = 1024;

    /*  Linear kernel */
    parameters.kernel_type = LINEAR;

    /* Some critical parameters */
    parameters.nr_weight = 0; // or number of elements in the array weight_label and weight.
        // parameters.weight_label;
        // parameters.weight;
    // for (parameters.C = 0.1; parameters.C <= 1; parameters.C += 1)
    // {

    //     error_msg = svm_check_parameter(&train_problem, &parameters);

    //     if (error_msg)
    //     {
    //         fprintf(stderr, "ERROR: %s\n", error_msg);
    //         exit(1);
    //     }

    //     struct svm_model *linear;

    //     printf("1. Linear model, C = %f\n", parameters.C);

    //     /* performance evaluation */
    //     do_cross_validation(&train_problem, &parameters, log_file);

    //     linear = svm_train(&train_problem, &parameters);

    //     /* predict results */
    //     double pred = 0, correct = 0;
    //     int confusion[5][4] = {};
    //     for (int i = 0; i < test_problem.l; i++)
    //     {
    //         pred = svm_predict(linear, test_problem.x[i]);
    //         if (pred == test_problem.y[i])
    //             correct++;
    //         for (int j = 1; j <= 5; j++)
    //         {
    //             /* positive at class-j */
    //             if (pred == j)
    //             {
    //                 /* TP at class-i */
    //                 if (j == test_problem.y[i])
    //                 {
    //                     confusion[j - 1][0]++;
    //                 }
    //                 else /* FP at class-i, FN at class-train_labels[k] */
    //                 {
    //                     confusion[j - 1][2]++;
    //                     confusion[(int)(test_problem.y[i] - 1)][1]++;
    //                 }
    //             }
    //             else
    //             {
    //                 if (j != test_problem.y[i])
    //                     confusion[j - 1][3]++;
    //             }
    //         }
    //     }

    //     printf("   test ACC: %f\n", correct / test_problem.l);
    //     for (int i = 0; i < 5; i++)
    //     { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
    //         printf("    %d, %d, %d, %d, %d, %f, %f\n",
    //                i,
    //                confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
    //                (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
    //                (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
    //     }

    //     FILE *sv_file = fopen("SV_linear.txt", "a+");
    //     int sv_count_start = 0;
    //     int sv_count_end = 0;
    //     for (int j = 0; j < 5; j++)
    //     {
    //         fprintf(sv_file, "SV of class %d: ", j);
    //         sv_count_end += linear->nSV[j];
    //         for (int i = sv_count_start; i < sv_count_end; i++)
    //         {
    //             fprintf(sv_file, "%d,", linear->sv_indices[i]);
    //         }
    //         fprintf(sv_file, "\n");
    //         sv_count_start += linear->nSV[j];
    //     }

    //     printf("\n");
    //     char model_file_name[1024];
    //     char C_value[10];
    //     strcpy(model_file_name, "Linear_model_");
    //     sprintf(C_value, "%f", parameters.C);
    //     strcat(model_file_name, C_value);

    //     // if (svm_save_model(model_file_name, linear))
    //     // {
    //     //     printf("can't save model to file %s\n", model_file_name);
    //     //     exit(1);
    //     // }
    //     svm_free_and_destroy_model(&linear);
    // }

    // /*  polynomial kernel */
    // parameters.kernel_type = POLY;

    // /* Some critical parameters */
    // parameters.nr_weight = 0; // or number of elements in the array weight_label and weight.
    //     // parameters.weight_label;
    //     // parameters.weight;
    // parameters.coef0 = 0.7;
    // for (parameters.C = 0.1; parameters.C <= 0.1; parameters.C += 0.3)
    // {
    //     for (parameters.degree = 2; parameters.degree <= 2; parameters.degree += 1)
    //     {
    //         for (parameters.gamma = 0.4; parameters.gamma <= 0.4; parameters.gamma += 0.1)
    //         {
    //             // for (parameters.coef0 = 0.1; parameters.coef0 <= 1.81; parameters.coef0 += 0.6)
    //             // {

    //             error_msg = svm_check_parameter(&train_problem, &parameters);

    //             if (error_msg)
    //             {
    //                 fprintf(stderr, "ERROR: %s\n", error_msg);
    //                 exit(1);
    //             }

    //             struct svm_model *poly;

    //             printf("    Poly model, %f - %d - %f - %f, ", parameters.C, parameters.degree, parameters.gamma, parameters.coef0);
    //             //do_cross_validation(&train_problem, &parameters, log_file);

    //             /* performance evaluation */
    //             do_cross_validation(&train_problem, &parameters, log_file);

    //             poly = svm_train(&train_problem, &parameters);

    //             /* predict results */
    //             double pred = 0, correct = 0;
    //             int confusion[5][4] = {};
    //             for (int i = 0; i < test_problem.l; i++)
    //             {
    //                 pred = svm_predict(poly, test_problem.x[i]);
    //                 if (pred == test_problem.y[i])
    //                     correct++;
    //                 for (int j = 1; j <= 5; j++)
    //                 {
    //                     /* positive at class-j */
    //                     if (pred == j)
    //                     {
    //                         /* TP at class-i */
    //                         if (j == test_problem.y[i])
    //                         {
    //                             confusion[j - 1][0]++;
    //                         }
    //                         else /* FP at class-i, FN at class-train_labels[k] */
    //                         {
    //                             confusion[j - 1][2]++;
    //                             confusion[(int)(test_problem.y[i] - 1)][1]++;
    //                         }
    //                     }
    //                     else
    //                     {
    //                         if (j != test_problem.y[i])
    //                             confusion[j - 1][3]++;
    //                     }
    //                 }
    //             }

    //             printf("     Test ACC: %f\n", correct / test_problem.l);
    //             for (int i = 0; i < 5; i++)
    //             { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
    //                 printf("    %d, %d, %d, %d, %d, %f, %f\n",
    //                        i,
    //                        confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
    //                        (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
    //                        (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
    //             }
    //             FILE *sv_file = fopen("SV_poly.txt", "a");
    //             int sv_count_start = 0;
    //             int sv_count_end = 0;
    //             for (int j = 0; j < 5; j++)
    //             {
    //                 fprintf(sv_file, "SV of class %d: ", j);
    //                 sv_count_end += poly->nSV[j];
    //                 for (int i = sv_count_start; i < sv_count_end; i++)
    //                 {
    //                     fprintf(sv_file, "%d,", poly->sv_indices[i]);
    //                 }
    //                 fprintf(sv_file, "\n");
    //                 sv_count_start += poly->nSV[j];
    //             }

    //             printf("\n");
    //             svm_free_and_destroy_model(&poly);
    //         }
    //     }
    // }

    // /*  RBF kernel */
    // parameters.kernel_type = RBF;

    // /* Some critical parameters */
    // parameters.nr_weight = 0; // or number of elements in the array weight_label and weight.
    // parameters.weight_label;
    // parameters.weight;
    // for (parameters.C = 0.4; parameters.C <= 0.4; parameters.C += 0.3)
    // {
    //     for (parameters.gamma = 0.01; parameters.gamma <= 0.1; parameters.gamma += 0.4)
    //     {

    //         error_msg = svm_check_parameter(&train_problem, &parameters);

    //         if (error_msg)
    //         {
    //             fprintf(stderr, "ERROR: %s\n", error_msg);
    //             exit(1);
    //         }

    //         printf("    RBF model, %f - %f, ", parameters.C, parameters.gamma);

    //         /* performance evaluation */
    //         do_cross_validation(&train_problem, &parameters, log_file);

    //         struct svm_model *RBF;

    //         RBF = svm_train(&train_problem, &parameters);

    //         /* predict results */
    //         double pred = 0, correct = 0;
    //         int confusion[5][4] = {};
    //         for (int i = 0; i < test_problem.l; i++)
    //         {
    //             pred = svm_predict(RBF, test_problem.x[i]);
    //             if (pred == test_problem.y[i])
    //                 correct++;
    //             for (int j = 1; j <= 5; j++)
    //             {
    //                 /* positive at class-j */
    //                 if (pred == j)
    //                 {
    //                     /* TP at class-i */
    //                     if (j == test_problem.y[i])
    //                     {
    //                         confusion[j - 1][0]++;
    //                     }
    //                     else /* FP at class-i, FN at class-train_labels[k] */
    //                     {
    //                         confusion[j - 1][2]++;
    //                         confusion[(int)(test_problem.y[i] - 1)][1]++;
    //                     }
    //                 }
    //                 else
    //                 {
    //                     if (j != test_problem.y[i])
    //                         confusion[j - 1][3]++;
    //                 }
    //             }
    //         }

    //         printf("    test ACC: %f\n", correct / test_problem.l);
    //         for (int i = 0; i < 5; i++)
    //         { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
    //             printf("    %d, %d, %d, %d, %d, %f, %f\n",
    //                    i,
    //                    confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
    //                    (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
    //                    (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
    //         }
    //         FILE *sv_file = fopen("SV_RBF.txt", "a");
    //         int sv_count_start = 0;
    //         int sv_count_end = 0;
    //         for (int j = 0; j < 5; j++)
    //         {
    //             fprintf(sv_file, "SV of class %d: ", j);
    //             sv_count_end += RBF->nSV[j];
    //             for (int i = sv_count_start; i < sv_count_end; i++)
    //             {
    //                 fprintf(sv_file, "%d ,", RBF->sv_indices[i]);
    //             }
    //             fprintf(sv_file, "\n");
    //             sv_count_start += RBF->nSV[j];
    //         }

    //         printf("\n");
    //         char model_file_name[1024];
    //         char C_value[10];
    //         strcpy(model_file_name, "RBF_model_");
    //         sprintf(C_value, "%.1f-%.1f", parameters.C, parameters.gamma);
    //         strcat(model_file_name, C_value);

    //         if (svm_save_model(model_file_name, RBF))
    //         {
    //             printf("can't save model to file %s\n", model_file_name);
    //             exit(1);
    //         }
    //         svm_free_and_destroy_model(&RBF);
    //     }
    // }

    /*  Linear + RBF kernel */
    parameters.kernel_type = PRECOMPUTED;
    parameters.shrinking = 0;

    /* Some critical parameters */
    parameters.nr_weight = 0; // or number of elements in the array weight_label and weight.
    // parameters.weight_label;
    // parameters.weight;
    double linear_w = 0, rbf_w = 0;

    printf("Precompute kernel get start!\n");

    for (linear_w = 0.5; linear_w < 1.6; linear_w += 0.5)
    {
        for (rbf_w = 0.5; rbf_w < 1.6; rbf_w += 0.5)
        {
            for (parameters.C = 0.4; parameters.C <= 0.4; parameters.C += 0.3)
            {
                for (parameters.gamma = 0.01; parameters.gamma <= 0.01; parameters.gamma += 0.4)
                {

                    error_msg = svm_check_parameter(&train_problem, &parameters);

                    if (error_msg)
                    {
                        fprintf(stderr, "ERROR: %s\n", error_msg);
                        exit(1);
                    }

                    /* precompute the value in file */
                    struct svm_problem precom_train_problem;
                    struct svm_problem precom_test_problem;

                    printf("Compute user-defined kernel value\n");

                    struct svm_node **X = (struct svm_node **)malloc(sizeof(struct svm_node *) * train_problem.l + sizeof(struct svm_node) * train_problem.l * (train_problem.l + 1));
                    if (!X)
                    {
                        printf("fail to malloc\n");
                    }
                    struct svm_node *pStruct;
                    int j = 0;
                    double precom_linear = 0, precom_rbf = 0;
                    for (j = 0, pStruct = (struct svm_node *)(X + train_problem.l); j < train_problem.l; j++, pStruct += (train_problem.l + 1))
                    {
                        X[j] = pStruct;
                    }
                    for (int i = 0; i < train_problem.l; i++)
                    {
                        printf("line %d ... ", i);
                        for (j = 0; j < (train_problem.l + 1); j++)
                        {

                            if (j == 0)
                            {
                                X[i][j].index = i;
                                X[i][j].value = 0;
                            }
                            else
                            {
                                X[i][j].index = j - 1;
                                precom_linear = 0;
                                precom_rbf = 0;
                                for (int k = 0; k < 784; k++)
                                {
                                    precom_linear += train_problem.x[i][k].value * train_problem.x[j - 1][k].value;
                                    precom_rbf += pow(train_problem.x[i][k].value - train_problem.x[j - 1][k].value, 2);
                                }
                                X[i][j].value = linear_w * precom_linear + rbf_w * exp(-parameters.gamma * precom_rbf);
                                //printf("%f\n", X[i][j].value);
                            }
                        }
                    }
                    precom_train_problem.x = X;
                    precom_train_problem.y = train_problem.y;
                    precom_train_problem.l = train_problem.l;
                    printf("Finish train data\n");
                    struct svm_node **X_test = (struct svm_node **)malloc(sizeof(struct svm_node *) * test_problem.l + sizeof(struct svm_node) * test_problem.l * (test_problem.l + 1));
                    if (!X_test)
                    {
                        printf("fail to malloc\n");
                    }
                    for (j = 0, pStruct = (struct svm_node *)(X + test_problem.l); j < test_problem.l; j++, pStruct += (test_problem.l + 1))
                    {
                        X_test[j] = pStruct;
                    }
                    for (int i = 0; i < test_problem.l; i++)
                    {
                        printf("line %d ... ", i);
                        for (j = 0; j < (test_problem.l + 1); j++)
                        {

                            if (j == 0)
                            {
                                X_test[i][j].index = i;
                                X_test[i][j].value = 0;
                            }
                            else
                            {
                                X_test[i][j].index = j - 1;
                                precom_linear = 0;
                                precom_rbf = 0;
                                for (int k = 0; k < 784; k++)
                                {
                                    precom_linear += test_problem.x[i][k].value * test_problem.x[j - 1][k].value;
                                    precom_rbf += pow(test_problem.x[i][k].value - test_problem.x[j - 1][k].value, 2);
                                }
                                X_test[i][j].value = linear_w * precom_linear + rbf_w * exp(-parameters.gamma * precom_rbf);
                            }
                        }
                    }
                    precom_test_problem.x = X_test;
                    precom_test_problem.y = test_problem.y;
                    precom_test_problem.l = test_problem.l;
                    printf("Finish test data\n");
                    free(train_problem.x);
                    free(train_problem.y);
                    free(test_problem.x);
                    free(test_problem.y);

                    struct svm_model *L_RBF;

                    printf("    L_RBF model, %f - %f - %f - %f, ", parameters.C, parameters.gamma, linear_w, rbf_w);

                    /* performance evaluation */
                    //do_cross_validation(&precom_train_problem, &parameters, log_file);

                    L_RBF = svm_train(&precom_train_problem, &parameters);
                    printf("Finish training\n");
                    /* predict results */
                    double pred = 0, correct = 0;
                    int confusion[5][4] = {};
                    for (int i = 0; i < precom_test_problem.l; i++)
                    {
                        pred = svm_predict(L_RBF, precom_test_problem.x[i]);
                        if (pred == precom_test_problem.y[i])
                            correct++;
                        for (int j = 1; j <= 5; j++)
                        {
                            /* positive at class-j */
                            if (pred == j)
                            {
                                /* TP at class-i */
                                if (j == precom_test_problem.y[i])
                                {
                                    confusion[j - 1][0]++;
                                }
                                else /* FP at class-i, FN at class-train_labels[k] */
                                {
                                    confusion[j - 1][2]++;
                                    confusion[(int)(precom_test_problem.y[i] - 1)][1]++;
                                }
                            }
                            else
                            {
                                if (j != precom_test_problem.y[i])
                                    confusion[j - 1][3]++;
                            }
                        }
                    }

                    printf("    Test ACC: %f\n", correct / precom_test_problem.l);
                    for (int i = 0; i < 5; i++)
                    { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
                        printf("    %d, %d, %d, %d, %d, %f, %f\n",
                               i,
                               confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
                               (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
                               (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
                    }

                    printf("\n");
                    char model_file_name[1024];
                    char C_value[10];
                    strcpy(model_file_name, "L_RBF_model_");
                    sprintf(C_value, "%.1f-%.1f", parameters.C, parameters.gamma);
                    strcat(model_file_name, C_value);

                    if (svm_save_model(model_file_name, L_RBF))
                    {
                        printf("can't save model to file %s\n", model_file_name);
                        exit(1);
                    }
                    svm_free_and_destroy_model(&L_RBF);

                    free(precom_test_problem.x);
                    free(precom_test_problem.y);
                    free(precom_train_problem.x);
                    free(precom_train_problem.y);
                }
            }
        }
    }

    svm_destroy_param(&parameters);
    free(train_problem.x);
    free(train_problem.y);

    return 0;
}
