#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"

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
    fprintf(fp, "Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / prob->l);
    free(target);
}

int main(int argc, char const *argv[])
{
    const char *error_msg;
    FILE *log_file = fopen("training_log.txt", "a+");
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
    for (parameters.C = 0.5; parameters.C <= 2.1; parameters.C += 0.1)
    {

        error_msg = svm_check_parameter(&train_problem, &parameters);

        if (error_msg)
        {
            fprintf(stderr, "ERROR: %s\n", error_msg);
            exit(1);
        }

        struct svm_model *linear;

        fprintf(log_file, "Linear model, %f, ", parameters.C);
        //do_cross_validation(&train_problem, &parameters, log_file);
        printf("Linear kernel\n");
        linear = svm_train(&train_problem, &parameters);

        /* predict results */
        double pred = 0, correct = 0;
        int confusion[5][4] = {};
        for (int i = 0; i < test_problem.l; i++)
        {
            pred = svm_predict(linear, test_problem.x[i]);
            if (pred == test_problem.y[i])
                correct++;
            for (int j = 1; j <= 5; j++)
            {
                /* positive at class-j */
                if (pred == j)
                {
                    /* TP at class-i */
                    if (j == test_problem.y[i])
                    {
                        confusion[j - 1][0]++;
                    }
                    else /* FP at class-i, FN at class-train_labels[k] */
                    {
                        confusion[j - 1][2]++;
                        confusion[(int)(test_problem.y[i] - 1)][1]++;
                    }
                }
                else
                {
                    if (j != test_problem.y[i])
                        confusion[j - 1][3]++;
                }
            }
        }

        fprintf(log_file, "%f, ", correct / test_problem.l);
        for (int i = 0; i < 5; i++)
        { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
            fprintf(log_file, "%d, %d, %d, %d, %d, %f, %f,",
                    i,
                    confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
                    (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
                    (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
        }
        fprintf(log_file, "\n");
        char model_file_name[1024];
        char C_value[10];
        strcpy(model_file_name, "Linear_model_C_");
        sprintf(C_value, "%f", parameters.C);
        strcat(model_file_name, C_value);

        if (svm_save_model(model_file_name, linear))
        {
            printf("can't save model to file %s\n", model_file_name);
            exit(1);
        }
        svm_free_and_destroy_model(&linear);
    }

    /*  polynomial kernel */
    parameters.kernel_type = POLY;

    /* Some critical parameters */
    parameters.nr_weight = 0; // or number of elements in the array weight_label and weight.
        // parameters.weight_label;
        // parameters.weight;
    for (parameters.C = 0.5; parameters.C <= 2.1; parameters.C += 0.1)
    {
        for (parameters.degree = 2; parameters.degree <= 10; parameters.degree += 1)
        {
            for (parameters.gamma = 0.5; parameters.gamma <= 2.1; parameters.degree += 0.1)
            {
                for (parameters.coef0 = 0.5; parameters.coef0 <= 2.1; parameters.coef0 += 0.1)
                {

                    error_msg = svm_check_parameter(&train_problem, &parameters);

                    if (error_msg)
                    {
                        fprintf(stderr, "ERROR: %s\n", error_msg);
                        exit(1);
                    }

                    struct svm_model *linear;

                    fprintf(log_file, "Linear model, %f, ", parameters.C);
                    //do_cross_validation(&train_problem, &parameters, log_file);
                    printf("Linear kernel\n");
                    linear = svm_train(&train_problem, &parameters);

                    /* predict results */
                    double pred = 0, correct = 0;
                    int confusion[5][4] = {};
                    for (int i = 0; i < test_problem.l; i++)
                    {
                        pred = svm_predict(linear, test_problem.x[i]);
                        if (pred == test_problem.y[i])
                            correct++;
                        for (int j = 1; j <= 5; j++)
                        {
                            /* positive at class-j */
                            if (pred == j)
                            {
                                /* TP at class-i */
                                if (j == test_problem.y[i])
                                {
                                    confusion[j - 1][0]++;
                                }
                                else /* FP at class-i, FN at class-train_labels[k] */
                                {
                                    confusion[j - 1][2]++;
                                    confusion[(int)(test_problem.y[i] - 1)][1]++;
                                }
                            }
                            else
                            {
                                if (j != test_problem.y[i])
                                    confusion[j - 1][3]++;
                            }
                        }
                    }

                    fprintf(log_file, "%f, ", correct / test_problem.l);
                    for (int i = 0; i < 5; i++)
                    { /*class i, TP, FN, FP, TN, Sensitivity, Specificity*/
                        fprintf(log_file, "%d, %d, %d, %d, %d, %f, %f,",
                                i,
                                confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3],
                                (double)confusion[i][0] / (double)(confusion[i][0] + confusion[i][1]),
                                (double)confusion[i][3] / (double)(confusion[i][3] + confusion[i][2]));
                    }
                    fprintf(log_file, "\n");
                    char model_file_name[1024];
                    char C_value[10];
                    strcpy(model_file_name, "Linear_model_C_");
                    sprintf(C_value, "%f", parameters.C);
                    strcat(model_file_name, C_value);

                    if (svm_save_model(model_file_name, linear))
                    {
                        printf("can't save model to file %s\n", model_file_name);
                        exit(1);
                    }
                    svm_free_and_destroy_model(&linear);
                }
            }
        }
    }

    svm_destroy_param(&parameters);
    free(train_problem.x);
    free(train_problem.y);

    return 0;
}
