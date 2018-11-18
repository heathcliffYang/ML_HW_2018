#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include "mnist_handlers.h"

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