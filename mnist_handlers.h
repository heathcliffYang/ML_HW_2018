#ifndef MNIST
#define MNIST
#include <inttypes.h>

static unsigned int mnist_bin_to_int(char *v);

int data_read(const char *labels_path, char *images_path, uint8_t **labels, uint8_t ***images, unsigned int *num_of_data, unsigned int *num_of_pixels);

#endif