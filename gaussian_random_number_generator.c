#include <stdlib.h>
#include "gaussian_random_number_generator.h"

double Marsaglia(double xi_mean, double xi_var)
{
    double x, y, s;
    do
    {
        x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        y = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = pow(x, 2) + pow(y, 2);
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    return xi_mean + sqrt(xi_var) * x * s;
};

/* Univariate gaussian data generator, outcome ~ N(mean, var) */
double Marsaglia_gaussian_rand_generator(double mean, double var)
{
    double N = 1000, x_bar = 0;
    for (int i = 0; i < N; i++)
    {
        x_bar += Marsaglia(mean, var * N);
    }
    return x_bar / N;
};

twoD_data_point twoD_data_generator(double mx, double vx, double my, double vy)
{
    twoD_data_point data_point;
    data_point.x = Marsaglia_gaussian_rand_generator(mx, vx);
    data_point.y = Marsaglia_gaussian_rand_generator(my, vy);

    return data_point;
};