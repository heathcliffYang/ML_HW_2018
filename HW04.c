#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian_random_number_generator.h"

int main(int argc, char **argv)
{
    /* mode 1 */
    if (atoi(argv[1]) == 1)
    {
        int num_of_data = atoi(argv[2]);

        /* argv [3 ~ 10]  */
        /* [mean][var] */
        double mean_var[4][2] = {};
        for (int i = 3; i <= 10; i++)
        {
            mean_var[(i - 3) / 2][1 * (i % 2 == 0)] = atof(argv[i]);
        }

        /* generate data */
        twoD_data_point *first_set = malloc();
        for (int i = 0; i < num_of_data; i++)
        {
            twoD_data_generator(mean_var[0][0], mean_var[0][1], mean_var[1][0], mean_var[1][1]);
        }
    }
    /* mode 2 */
    else
    {
        ;
    }

    return 0;
}