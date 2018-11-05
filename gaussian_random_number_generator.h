#ifndef GAUSSIAN_RAND_NUM_GEN
#define GAUSSIAN_RAND_NUM_GEN

double Marsaglia(double xi_mean, double xi_var);

double Marsaglia_gaussian_rand_generator(double mean, double var);

typedef struct twoD_data_point
{
    double x;
    double y;
} twoD_data_point;

/* x and y are independently sampled */
twoD_data_point twoD_data_generator(double mx, double vx, double my, double vy);

#endif