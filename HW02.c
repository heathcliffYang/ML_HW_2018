#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>

char input_dir[] = "/Users/ginny0922/Desktop/NCTU/ML2018-1/HW/HW2_input/";
char train_labels_input[] = "train-labels.idx1-ubyte";
char train_images_input[] = "train-images.idx3-ubyte";
char test_labels_input[] = "t10k-labels.idx1-ubyte";
char test_images_input[] = "t10k-images.idx3-ubyte";

/*
typedef struct mnist_data {
    uint8_t label;
    uint8_t pixels[28][28];
} mnist_data;
*/

/* Source from https://github.com/projectgalateia/mnist/blob/master/mnist.h */
/* Flip the bytes */
static unsigned int mnist_bin_to_int(char *v)
{
    int i;
    unsigned int ret = 0;

    for (i = 0; i < 4; ++i)
    {
        ret <<= 8;
        /* Unsigned hexadecimal integer */
        /* printf("%hhx\n", v[i]); */
        ret |= (unsigned char)v[i];
    }

    return ret;
};

void initialize(double **array, int num)
{
    int i = 0;
    for (i = 0; i < num; i++)
    {
        (*array)[i] = 0;
    }
};

/* MNIST data read */
int data_read(const char *labels_path, char *images_path, uint8_t **labels, uint8_t ***images, unsigned int *num_of_data, unsigned int *num_of_pixels)
{
    FILE *fp_labels = fopen(labels_path, "r");
    FILE *fp_images = fopen(images_path, "r");

    if (fp_labels == NULL)
    {
        printf("Fail to open the label-input file\n");
        return -1;
    }
    if (!fp_images)
    {
        printf("Fail to open the image-input file\n");
        return -1;
    }

    /* Extract basic variables: # of data, # of pixels an images owns */
    char buf[4];
    fread(buf, 1, 4, fp_labels);
    if (mnist_bin_to_int(buf) != 2049)
    {
        printf("This isn't training data Labels\n");
        return -1;
    }
    fread(buf, 1, 4, fp_labels);
    *num_of_data = mnist_bin_to_int(buf);
    printf("This label-input file contains %u labels\n", *num_of_data);

    fread(buf, 1, 4, fp_images);
    if (mnist_bin_to_int(buf) != 2051)
    {
        printf("This isn't training data Images\n");
        return -1;
    }
    fread(buf, 1, 4, fp_images);
    fread(buf, 1, 4, fp_images);
    *num_of_pixels = mnist_bin_to_int(buf);
    fread(buf, 1, 4, fp_images);
    *num_of_pixels *= mnist_bin_to_int(buf);
    printf("There are %d labels/images with %d pixels in this pair of dataset.\n",
           *num_of_data, *num_of_pixels);

    /* malloc 2-D array for images, 1-D one for labels */
    printf("Start to malloc 2-D array for images, 1-D one for labels\n");
    (*labels) = (uint8_t *)malloc(sizeof(uint8_t) * *num_of_data);
    (*images) = (uint8_t **)malloc(sizeof(uint8_t *) * *num_of_data + *num_of_data * *num_of_pixels * sizeof(uint8_t));
    uint8_t *pData;
    int j = 0;
    for (j = 0, pData = (uint8_t *)(*images + *num_of_data); j < *num_of_data; j++, pData += *num_of_pixels)
    {
        (*images)[j] = pData;
    }

    printf("Start to read ...\n");
    /* Read labels & images */
    if (*num_of_data != fread((*labels), 1, *num_of_data, fp_labels))
    {
        printf("Miss to read labels\n");
    }
    int num_of_data_block = 600;
    int iter = *num_of_data / num_of_data_block;
    for (int i = 0; i < iter; i++)
    {
        if (feof(fp_images) || ferror(fp_images) || feof(fp_labels) || ferror(fp_labels))
            break;
        if (num_of_data_block * *num_of_pixels != fread((*images)[i * num_of_data_block], 1, num_of_data_block * *num_of_pixels, fp_images))
        {
            printf("Read miss at %d iteration\n", i);
            return -1;
        }
    }

    printf("End read\n");
    fclose(fp_images);
    fclose(fp_labels);
    printf("Close these files\n");
    return 0;
};

double gaussian_log(double *mean, double *var, double *value)
{
    double m = -pow((*value - *mean), 2) / (2 * *var) - log(2 * 3.14 * *var) / 2;
    //printf("value: %f, mean: %f, var: %f, m: %f\n", (double) *value, *mean, *var, m);
    return m;
};

int generate_binary_outcomes(char *filepath, int num_of_lines)
{
    int i = 0, j = 0, outcomes_per_lines = 0, outcome = 0, fair = 0;
    FILE *fp = fopen(filepath, "w+");
    if (!fp)
    {
        printf("Cannot write the binary outcomes\n");
        return -1;
    }
    printf("Generating the binary outcome file\n");
    srand(time(NULL));
    for (i = 0; i < num_of_lines; i++)
    {
        outcomes_per_lines = rand() % 10;
        fair = rand() & 3;
        for (j = 0; j < outcomes_per_lines; j++)
        {
            if (fair <= 2)
            {
                outcome = rand() & 1;
                fprintf(fp, "%d", outcome);
            }
            else /* NOT Fair */
            {
                if ((rand() & 15) == 7)
                    fprintf(fp, "0");
                else
                    fprintf(fp, "1");
            }
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
};

double binomial_likelihood(int N, int m, double p)
{
    double likelihood = 1;
    int i = 0, j = 0, c = 0;
    /* N-m times*/
    for (i = N, j = (N - m), c = 0; i >= (m + 1) && j >= 1; i--, j--, c++)
    {
        likelihood *= (1 - p) * i / j;
        if (c < m)
        {
            likelihood *= p;
        }
    }
    while (c < m)
    {
        likelihood *= p;
        c++;
    }
    return likelihood;
};

int beta_distribution_online_learning(FILE *fp, int initial_a, int initial_b)
{
    /* Beta distribution, a for times of 1 */
    double likelihood = 0, a = (double)initial_a, b = (double)initial_b, m = 0, i = 0;
    int N = 0, count = 0;
    char line[1001] = {};
    /* Read per line, which is a round of judgement */
    while (fgets(line, 1001, fp) != NULL)
    {
        count++;
        while (line[N] != '\n')
        {
            if (line[N] == '1')
            {
                m++;
            }
            else if (line[N] != '0')
            {
                printf("Read error\n");
                return -1;
            }
            N++;
        }
        printf("\nInput %d : total: %d, [1]: %.f, [0]: %.f\n", count, N, m, N - m);
        if ((a + b) <= 2)
            printf("Beta prior's: a = %.f, b = %.f, p = %f\n", a, b, (a) / (a + b));
        else
            printf("Beta prior's: a = %.f, b = %.f, p = %f\n", a, b, (a - 1) / (a + b - 2));
        likelihood = binomial_likelihood(N, m, (a - 1) / (a + b - 2));
        printf("likelihood: %f\n", likelihood);
        /* Posterior */
        a += m;
        b += (N - m);
        printf("Beta posterior's: a = %.f, b = %.f; P_MAP = %f\n\n", a, b, (double)(a - 1) / (a + b - 2));
        N = 0;
        m = 0;
    }
    return 0;
};

int main(int argc, char **argv)
{
    if (atoi(argv[1]) < 2)
    {
        /* argv[1] = 0(discrete), 1(continuous), 2(online learning) */
        /* Read training data Labels */
        char train_label_path[77] = "", train_image_path[77] = "", test_label_path[76] = "", test_image_path[76] = "";
        strcat(train_label_path, input_dir);
        strcat(train_image_path, input_dir);
        strcat(train_label_path, train_labels_input);
        strcat(train_image_path, train_images_input);
        strcat(test_label_path, input_dir);
        strcat(test_image_path, input_dir);
        strcat(test_label_path, test_labels_input);
        strcat(test_image_path, test_images_input);

        uint8_t *train_labels = NULL, **train_images = NULL, *test_labels = NULL, **test_images = NULL;
        int read_success = 0, i = 0, j = 0, k = 0, x = 0;
        unsigned int train_num_of_data = 0, train_num_of_pixels = 0, test_num_of_data = 0, test_num_of_pixels = 0;
        read_success = data_read(train_label_path, train_image_path, &train_labels, &train_images, &train_num_of_data, &train_num_of_pixels);
        read_success *= data_read(test_label_path, test_image_path, &test_labels, &test_images, &test_num_of_data, &test_num_of_pixels);

        /* Check my input?? */
        if (read_success == -1)
        {
            printf("Fail to read files\n");
        }

        /* 1. Naive Bayes classifier */
        if (atoi(argv[1]) == 0)
        {

            /* -- Discrete */
            /* Prior's numerator = # of label i & Likelihood cube = label * pixels * F-value */
            printf("Discrete version\nBuild Prior's numerator & Likelihood cube\n");
            double likelihood_cube[10][784][32] = {};
            /* Add pseudocount */
            for (i = 0; i < 32; i++)
            {
                for (j = 0; j < train_num_of_pixels; j++)
                {
                    for (k = 0; k < 10; k++)
                    {
                        likelihood_cube[k][j][i] = 1;
                    }
                }
            }

            double prior_numerator[10] = {};
            for (i = 0; i < 10; i++)
            {
                prior_numerator[i] = 320;
            }
            for (i = 0; i < train_num_of_data; i++)
            {
                prior_numerator[train_labels[i]]++;
                for (j = 0; j < train_num_of_pixels; j++)
                {
                    likelihood_cube[train_labels[i]][j][train_images[i][j] >> 3]++;
                }
            }

            /* Testing */
            /* Compute a test image with each label likelihood and prior */
            double posterior = 0, max = -INFINITY, marginal_term = 0, likelihood_term = 0, num_of_error = 0;

            /* predict a possible label for each image */
            int *predictions = (int *)malloc(sizeof(int) * test_num_of_data);

            /* marginal of this test image (each pixel) */
            double *marginal = (double *)malloc(sizeof(double) * test_num_of_pixels);
            initialize(&marginal, test_num_of_pixels);

            printf("\nStart testing...\n");
            for (i = 0; i < test_num_of_data; i++)
            {
                /* marginal term */
                for (j = 0; j < test_num_of_pixels; j++)
                {
                    /* compute marginal of pixel j, running on label = (# of Di=Fi) */
                    for (k = 0; k < 10; k++)
                    {
                        marginal[j] += likelihood_cube[k][j][test_images[i][j] >> 3];
                    }
                    //printf("marginal[%d]=%f -> ", j, marginal[j]);
                    if (marginal[j] == 0)
                    {
                        printf("Oops!\n");
                        return -1;
                    }
                    marginal_term += log(marginal[j]);
                    //printf("marginal term : %f\n", marginal_term);
                }

                /* compute posterior for each label */
                printf("Posterior of test image %d:(truth: %d)\n\b\b", i, test_labels[i]);
                for (x = 0; x < 10; x++)
                {
                    /* likelihood term for label x */
                    for (j = 0; j < test_num_of_pixels; j++)
                    {
                        /* compute likelihood of pixel i in label x */
                        likelihood_term += log(likelihood_cube[x][j][test_images[i][j] >> 3]);
                    }
                    //printf("likelihood_term %f\n", likelihood_term);
                    /* posterior */
                    posterior = likelihood_term - marginal_term + (test_num_of_pixels - 1) * log((train_num_of_data + 3200) / prior_numerator[x]);
                    //printf("(%.2f - %.2f + %.2f )= ", likelihood_term, marginal_term, (test_num_of_pixels - 1) * log((train_num_of_data+3200) / prior_numerator[x]));
                    if (max < posterior)
                    {
                        predictions[i] = x;
                        max = posterior;
                    }
                    printf("%.2f ", posterior);
                    likelihood_term = 0;
                }
                marginal_term = 0;
                initialize(&marginal, test_num_of_pixels);
                max = -INFINITY;
                printf("predict: %d\n", predictions[i]);
                if (predictions[i] != test_labels[i])
                {
                    num_of_error++;
                }
            }

            printf("Error rate is %f\n", num_of_error / test_num_of_data);

            printf("\nPrior from 0 to 9\n");
            for (i = 0; i < 9; i++)
            {
                printf("%f, ", prior_numerator[i] / (train_num_of_data + 3200));
            }
            printf("%f\n", prior_numerator[9] / (train_num_of_data + 3200));

            free(predictions);
            free(marginal);
        }
        else if (atoi(argv[1]) == 1)
        {
            /* -- Continuous */
            printf("Continuous version\n");
            /* MLE on Gaussian: calculate the value in each label's i-th pixel: mean & variance */
            double gaussian_model[10][784][2] = {}, prior[10] = {}, max = -INFINITY;
            int *predictions = (int *)malloc(sizeof(int) * test_num_of_data);

            for (i = 0; i < train_num_of_data; i++)
            {
                prior[train_labels[i]]++;
                for (j = 0; j < train_num_of_pixels; j++)
                {
                    /* mean's numerator */
                    gaussian_model[train_labels[i]][j][0] += train_images[i][j];
                    /* E(X^2) */
                    gaussian_model[train_labels[i]][j][1] += pow(train_images[i][j], 2);
                }
            }
            //FILE *fp_mean = fopen("/Users/ginny0922/Desktop/NCTU/ML2018-1/HW/HW2_input/gaussian_mean.csv", "w+");
            //FILE *fp_var = fopen("/Users/ginny0922/Desktop/NCTU/ML2018-1/HW/HW2_input/gaussian_var.csv", "w+");
            for (i = 0; i < train_num_of_pixels; i++)
            {
                for (j = 0; j < 10; j++)
                {
                    gaussian_model[j][i][0] /= prior[j];
                    gaussian_model[j][i][1] /= prior[j];
                    gaussian_model[j][i][1] -= pow(gaussian_model[j][i][0], 2);
                    //gaussian_model[j][i][1] = 1;
                    //fprintf(fp_mean, "%.1f,", gaussian_model[j][i][0]);
                    //fprintf(fp_var, "%.1f,", gaussian_model[j][i][1]);
                }
                //fprintf(fp_mean, "\n");
                //fprintf(fp_var, "\n");
            }
            int count = 0;
            double posterior[10] = {}, error_rate = 0, marginal = 0;
            for (i = 0; i < test_num_of_data; i++)
            {
                //printf("Posterior of test image %d:(truth: %d)\n\b\b", i, test_labels[i]);
                for (k = 0; k < 10; k++)
                {
                    for (j = 0; j < test_num_of_pixels; j++)
                    {
                        if (test_images[i][j] < 62 && gaussian_model[k][j][0] < 62)
                            posterior[k] += 1;
                        else if (gaussian_model[k][j][1] < 1 && test_images[i][j] != gaussian_model[k][j][0])
                            //else if (gaussian_model[k][j][1] < 1 && fabs(test_images[i][j] - gaussian_model[k][j][0]) > 10)
                            posterior[k] += 0;
                        else
                        {
                            for (double x = test_images[i][j]; x < test_images[i][j] + 1; x += 0.1)
                            {
                                //printf("Cal.... %f, %d\n", x, test_images[i][j]+1);
                                posterior[k] += gaussian_log(&gaussian_model[k][j][0], &gaussian_model[k][j][1], &x);
                                count++;
                            }
                        }
                    }
                    posterior[k] += (log(prior[k]) - log(train_num_of_data));
                    posterior[k] /= count;
                    count = 0;

                    marginal += posterior[k];
                    //printf("%.2f ", posterior);
                    if (max < posterior[k])
                    {
                        max = posterior[k];
                        //printf("change to %d\n", k);
                        predictions[i] = k;
                    }
                }
                for (x = 0; x < 10; x++)
                {
                    //printf("%.2f ", posterior[x] - marginal);
                    posterior[x] = 0;
                }
                //printf("predict: %d\n", predictions[i]);
                marginal = 0;
                max = -INFINITY; // Fuck you!!! I forgot to clear this compared value. Shit.
            }

            for (i = 0; i < test_num_of_data; i++)
            {
                if (predictions[i] != test_labels[i])
                {
                    error_rate++;
                }
            }
            printf("Error rate is %f\n", error_rate / test_num_of_data);
            free(predictions);
        }
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
    }
    else if (atoi(argv[1]) == 2)
    {
        /* 2. Online learning  */
        FILE *fp = fopen(argv[2], "r");
        if (!fp)
        {
            printf("Cannot open the binary outcomes file\n");
        }
        printf("Online learning - beta_distribution\n\n");

        if (beta_distribution_online_learning(fp, atoi(argv[3]), atoi(argv[4])) == -1)
        {
            printf("Failed\n");
        }
    }
    else if (atoi(argv[1]) == 3)
    {
        /* Filename & num_of_lines */
        generate_binary_outcomes(argv[2], atoi(argv[3]));
    }
    return 0;
}