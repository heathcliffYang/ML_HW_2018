#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <SymEigsSolver.h>

using namespace Spectra;
using namespace std;

// typedef struct data_point
// {
//     double x;
//     double y;
//     int cluster;
// } data_point;

// typedef struct cluster_center
// {
//     double x;
//     double y;
//     int num_of_element;
// } cluster_center;

// /* data points record */
// vector<data_point> data_points;

// vector<cluster_center> centers;
double X_train[5000][784] = {};
double X_test[2500][784] = {};
double X_train_c[784] = {};
double X_test_c[784] = {};
Eigen::MatrixXd covariance_train(784, 784);
Eigen::MatrixXd covariance_test(784, 784);

int reader(int data_mode)
{
    string line;
    string file_name;
    if (data_mode == 0)
        file_name = "X_train.txt";
    else
        file_name = "X_test.txt";
    ifstream input_file(file_name);

    if (input_file.is_open())
    {
        if (data_mode == 0)
        {
            for (int j = 0; j < 5000; j++)
            {
                for (int i = 0; i < 784 - 1; i++)
                {
                    getline(input_file, line, ',');
                    X_train[j][i] = stod(line);
                }
                getline(input_file, line);
                X_train[j][783] = stod(line);
            }
        }
        else
        {
            for (int j = 0; j < 2500; j++)
            {
                for (int i = 0; i < 784 - 1; i++)
                {
                    getline(input_file, line, ',');
                    X_test[j][i] = stod(line);
                }
                getline(input_file, line);
                X_test[j][783] = stod(line);
            }
        }
        input_file.close();
        cout << "End read." << endl;
    }
    else
    {
        cout << "Cannot open" << endl;
        return -1;
    }
    return 0;
};

int PCA()
{
    /* compute centers */
    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 5000; j++)
        {
            X_train_c[i] += X_train[j][i];
        }
        X_train_c[i] /= 5000;
    }
    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 2500; j++)
        {
            X_test_c[i] += X_test[j][i];
        }
        X_test_c[i] /= 2500;
    }

    double tmp_x[784] = {};

    for (int i = 0; i < 5000; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            tmp_x[j] = X_train[i][j] - X_train_c[j];
        }

        for (int j = 0; j < 784; j++)
        {
            for (int k = 0; k < 784; k++)
            {
                covariance_train(j, k) += tmp_x[j] * tmp_x[k] / 5000;
            }
        }
    }

    for (int i = 0; i < 2500; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            tmp_x[j] = X_test[i][j] - X_test_c[j];
        }

        for (int j = 0; j < 784; j++)
        {
            for (int k = 0; k < 784; k++)
            {
                covariance_test(j, k) += tmp_x[j] * tmp_x[k] / 2500;
            }
        }
    }
    DenseSymMatProd<double> op_train(covariance_train);
    DenseSymMatProd<double> op_test(covariance_test);
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs_train(&op_train, 2, 2 * (2 + 1));
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs_test(&op_test, 2, 2 * (2 + 1));
    eigs_train.init();
    eigs_test.init();
    eigs_train.compute();
    eigs_test.compute();
    // Retrieve results
    Eigen::VectorXd evalues_train;
    Eigen::MatrixXd evectors_train;
    if (eigs_train.info() == SUCCESSFUL)
    {
        evalues_train = eigs_train.eigenvalues();
        evectors_train = eigs_train.eigenvectors();
    }
    // Retrieve results
    Eigen::VectorXd evalues_test;
    Eigen::MatrixXd evectors_test;
    if (eigs_test.info() == SUCCESSFUL)
    {
        evalues_test = eigs_test.eigenvalues();
        evectors_test = eigs_test.eigenvectors();
    }
};

int main()
{
    reader(0);
    reader(1);

    return 0;
}