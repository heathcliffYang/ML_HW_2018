#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <SymEigsSolver.h>
#include <Eigen/QR>

using namespace Spectra;
using namespace std;

double X_train[5000][784] = {0};
double X_test[2500][784] = {0};
double X_train_c[784] = {0};
double X_test_c[784] = {0};
double pca_2d_train[5000][2] = {0};
double pca_2d_test[2500][2] = {0};
int T_train[5000] = {0};
int T_test[2500] = {0};

int reader(int data_mode)
{
    string line;
    string file_name;
    if (data_mode == 0)
        file_name = "X_train.csv";
    else
        file_name = "X_test.csv";
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
            /* compute centers */
            for (int i = 0; i < 784; i++)
            {
                for (int j = 0; j < 5000; j++)
                {
                    X_train_c[i] += X_train[j][i];
                }
                X_train_c[i] /= 5000;
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
            for (int i = 0; i < 784; i++)
            {
                for (int j = 0; j < 2500; j++)
                {
                    X_test_c[i] += X_test[j][i];
                }
                X_test_c[i] /= 2500;
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

double Kernel(int kernel_choose, int data_num, int i, int j, double gamma)
{
    double tmp = 0;
    if (data_num == 5000)
    {
        switch (kernel_choose)
        {
        case 1: // RBF
            for (int k = 0; k < 784; k++)
            {
                tmp += pow(X_train[i][k] - X_train[j][k], 2);
            }
            return exp(-gamma * tmp);
        case 2: // linear
            for (int k = 0; k < 784; k++)
            {
                tmp += X_train[i][k] * X_train[j][k];
            }
            return tmp;
        default:
            double tmp_1 = 0;
            for (int k = 0; k < 784; k++)
            {
                tmp += pow(X_train[i][k] - X_train[j][k], 2);
            }
            for (int k = 0; k < 784; k++)
            {
                tmp_1 += X_train[i][k] * X_train[j][k];
            }
            return exp(-gamma * tmp) + tmp_1;
        }
    }
    else
    {
        switch (kernel_choose)
        {
        case 1: // RBF
            for (int k = 0; k < 784; k++)
            {
                tmp += pow(X_test[i][k] - X_test[j][k], 2);
            }
            return exp(-gamma * tmp);
        case 2: // linear
            for (int k = 0; k < 784; k++)
            {
                tmp += X_test[i][k] * X_test[j][k];
            }
            return tmp;
        default:
            double tmp_1 = 0;
            for (int k = 0; k < 784; k++)
            {
                tmp += pow(X_test[i][k] - X_test[j][k], 2);
            }
            for (int k = 0; k < 784; k++)
            {
                tmp_1 += X_test[i][k] * X_test[j][k];
            }
            return exp(-gamma * tmp) + tmp_1;
        }
    }
};

int PCA(int write_file, int kernel_choose, int data_num, double gamma)
{

    if (kernel_choose == 3)
    {
        double tmp_x[784] = {};
        Eigen::MatrixXd covariance_train(784, 784);
        Eigen::MatrixXd covariance_test(784, 784);
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

        cout << "compute pca's eigen problem\n";

        eigs_train.compute();
        eigs_test.compute();

        cout << "end\n";

        //Retrieve results
        Eigen::VectorXd evalues_train;
        Eigen::MatrixXd evectors_train;
        if (eigs_train.info() == SUCCESSFUL)
        {
            evalues_train = eigs_train.eigenvalues();
            evectors_train = eigs_train.eigenvectors();
        }

        ofstream ouput_train_2D("PCA_train.txt", ios::app);

        for (int i = 0; i < 5000; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                pca_2d_train[i][0] += X_train[i][j] * evectors_train(j, 0);
                pca_2d_train[i][1] += X_train[i][j] * evectors_train(j, 1);
            }
            if (write_file == 1)
            {
                ouput_train_2D << pca_2d_train[i][0] << "," << pca_2d_train[i][1] << "\n";
            }
        }
        ouput_train_2D.close();

        // Retrieve results
        Eigen::VectorXd evalues_test;
        Eigen::MatrixXd evectors_test;
        if (eigs_test.info() == SUCCESSFUL)
        {
            evalues_test = eigs_test.eigenvalues();
            evectors_test = eigs_test.eigenvectors();
        }
        cout << "write test\n";
        ofstream ouput_test_2D("PCA_test.txt", ios::app);
        for (int i = 0; i < 2500; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                pca_2d_test[i][0] += X_test[i][j] * evectors_test(j, 0);
                pca_2d_test[i][1] += X_test[i][j] * evectors_test(j, 1);
            }
            if (write_file == 1)
            {
                ouput_test_2D << pca_2d_test[i][0] << "," << pca_2d_test[i][1] << "\n";
            }
        }
        ouput_test_2D.close();
        cout << "end\n";
    }
    else
    {
        Eigen::MatrixXd covariance(data_num, data_num);
        Eigen::MatrixXd ones(data_num, data_num);
        cout << "Start PCA projection " << data_num << " kernel is " << kernel_choose << endl;
        for (int i = 0; i < data_num; i++)
        {
            for (int j = 0; j < data_num; j++)
            {
                covariance(i, j) = 0;
            }
        }
        for (int i = 0; i < data_num; i++)
        {
            for (int j = 0; j < data_num; j++)
            {
                covariance(i, j) += Kernel(kernel_choose, data_num, i, j, gamma) / data_num;
                ones(i, j) = 1 / data_num;
            }
        }

        Eigen::MatrixXd covariance_c = covariance - ones * covariance - covariance * ones + ones * covariance * ones;

        DenseSymMatProd<double> op(covariance_c);
        SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, 2, 2 * (2 + 1));
        eigs.init();

        cout << "compute pca's eigen problem\n";

        eigs.compute();

        cout << "end\n";

        //Retrieve results
        Eigen::VectorXd evalues;
        Eigen::MatrixXd evectors;
        if (eigs.info() == SUCCESSFUL)
        {
            evalues = eigs.eigenvalues();
            evectors = eigs.eigenvectors();
        }
        int g = (int)gamma;

        Eigen::MatrixXd project = covariance * evectors;

        ofstream ouput_train_2D("kernel_pca/PCA_k_" + to_string(kernel_choose) + "_d_" + to_string(data_num) + "_g_" + to_string(g) + ".txt", ios::app);

        for (int i = 0; i < data_num; i++)
        {

            pca_2d_train[i][0] = project(i, 0);
            pca_2d_train[i][1] = project(i, 1);

            if (write_file == 1)
            {
                ouput_train_2D << pca_2d_train[i][0] << "," << pca_2d_train[i][1] << "\n";
            }
        }
        ouput_train_2D.close();

        cout << "end\n";
    }
    return 0;
};

bool is_converge(vector<vector<double>> centers, int num_of_cl)
{
    /* Check out if centers become steady */
    static double old_centers[5][5];
    double error_of_a_center = 0, error = 0;
    // cout << "converge check\n";

    for (int i = 0; i < num_of_cl; i++)
    {
        for (int j = 0; j < num_of_cl; j++)
        {

            error_of_a_center += pow(old_centers[i][j] - centers[i][j], 2);
            old_centers[i][j] = centers[i][j];
            //cout << centers[i][j] << ",";
        }
        //cout << endl;
        error += sqrt(error_of_a_center);
        error_of_a_center = 0;
    }
    cout << "Error is " << error / (double)num_of_cl << "\n";
    if (error / (double)num_of_cl < 0.00000001)
    {
        return false;
    }
    return true;
};

int k_means(vector<vector<double>> U, int data_num, int num_of_cl, int kernel_choose, int spec_mode, double gamma)
{
    cout << "k-means start\n";
    /* init centers */
    vector<vector<double>> centers(num_of_cl);
    vector<int> cl_index(data_num);
    vector<int> cl_elements(num_of_cl);
    double tmp = 0, min = 0;
    int belong_index = 0, iter = 0;
    cout << "Start initialization\n";
    for (int i = 0; i < num_of_cl; i++)
    {
        cl_elements[i] = 0;
        centers[i] = vector<double>(num_of_cl);
        for (int j = 0; j < num_of_cl; j++)
            centers[i][j] = 0;
    }
    for (int i = 0; i < data_num; i++)
    {
        cl_index[i] = rand() % num_of_cl;
        cl_elements[cl_index[i]]++;
        for (int j = 0; j < num_of_cl; j++)
            centers[cl_index[i]][j] += U[i][j];
    }
    for (int i = 0; i < num_of_cl; i++)
        for (int j = 0; j < num_of_cl; j++)
            centers[i][j] /= cl_elements[i];
    // for (int i = 0; i < num_of_cl; i++)
    // {

    //     for (int j = 0; j < num_of_cl; j++)
    //     {
    //         centers[i][j] = (rand() / (double)RAND_MAX) * 0.002 - 1.0;
    //     }
    // }
    cout
        << "Finish initialization\n";
    do
    {
        iter++;
        cout << "\niter " << iter << endl;
        for (int i = 0; i < data_num; i++)
        {
            for (int j = 0; j < num_of_cl; j++)
            {
                // cout << "to " << j << "-th center\n";
                for (int k = 0; k < num_of_cl; k++)
                {
                    tmp += pow(U[i][k] - centers[j][k], 2);
                }
                if (tmp != 0)
                {
                    tmp = sqrt(tmp);
                }

                if (min > tmp)
                {
                    min = tmp;
                    belong_index = j;
                    // cout << "belong index changes to " << j << endl;
                }
                // cout << "distance b/w" << i << " and " << j << " is " << tmp << endl;
                tmp = 0;
            }
            cl_index[i] = belong_index;
            cl_elements[belong_index]++;
            min = std::numeric_limits<double>::max();
        }

        for (int j = 0; j < num_of_cl; j++)
        {
            for (int k = 0; k < num_of_cl; k++)
            {
                centers[j][k] = 0;
            }
        }

        for (int i = 0; i < data_num; i++)
        {
            for (int j = 0; j < num_of_cl; j++)
            {
                centers[cl_index[i]][j] += U[i][j];
            }
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            cout << "Cluster " << i << " contains " << cl_elements[i] << endl;
            for (int j = 0; j < num_of_cl; j++)
            {
                if (cl_elements[i] != 0)
                    centers[i][j] /= cl_elements[i];
            }
            cl_elements[i] = 0;
        }

    } while (iter < 10 && is_converge(centers, num_of_cl));

    cout << "k-means finish\n";
    int g = (int)gamma;

    if (spec_mode == 0)
    {
        ofstream output_file("kernel_pca/ratio_cut_k_" + to_string(kernel_choose) + "_d_" + to_string(data_num) + "_g_" + to_string(g) + ".csv", ios::app);

        for (int i = 0; i < data_num; i++)
        {
            output_file << cl_index[i] << '\n';
        }
        output_file.close();
    }
    else
    {
        ofstream output_file("kernel_pca/normalized_cut_k_" + to_string(kernel_choose) + "_d_" + to_string(data_num) + "_g_" + to_string(g) + ".csv", ios::app);

        for (int i = 0; i < data_num; i++)
        {
            output_file << cl_index[i] << '\n';
        }
        output_file.close();
    }

    return 0;
};

int Spectral(int kernel_choose, int num_of_cl, int data_num, double gamma)
{
    cout << "Spectral clustering start\n"
         << "data_num: " << data_num << " gamma: " << gamma << " kernel " << kernel_choose << endl;

    vector<vector<double>> W(data_num);

    /* compute W according to different kernel */
    for (int i = 0; i < data_num; i++)
    {
        cout << i << "... ";
        W[i] = vector<double>(data_num);
        for (int j = 0; j < data_num; j++)
        {
            W[i][j] = Kernel(kernel_choose, data_num, i, j, gamma);
        }
    }

    Eigen::MatrixXd L(data_num, data_num);
    vector<double> D(data_num);
    /* compute L and D (D is for inverse later) */
    for (int i = 0; i < data_num; i++)
    {
        D[i] = 0;
    }
    for (int i = 0; i < data_num; i++)
    {
        L(i, i) = 0;
        for (int j = 0; j < data_num; j++)
        {
            D[i] += W[i][j];
            if (i != j)
            {
                L(i, j) = -W[i][j];
                L(i, i) += W[i][j];
            }
        }
    }

    W.clear();

    cout << "Ratio Cut\n";
    int g = (int)gamma;
    ofstream output_file("ratio_cut_data_point_k_" + to_string(kernel_choose) + "_d_" + to_string(data_num) + "_g_" + to_string(g) + ".csv", ios::app);

    /* L for ratio cut */
    DenseSymMatProd<double> op(L);
    SymEigsSolver<double, SMALLEST_ALGE, DenseSymMatProd<double>> eigs(&op, 5 + 1, 2 * (5 + 1 + 1));
    eigs.init();

    cout << "compute eigenvectors\n";
    eigs.compute();
    cout << "finished\n";
    vector<vector<double>> U(data_num);
    Eigen::MatrixXd u;
    if (eigs.info() == SUCCESSFUL)
    {
        u = eigs.eigenvectors();
    }
    else
    {
        cout << "Fail to compute eigenvectors\n";
        return -1;
    }

    cout << "Put eigenvectors in U\n";

    /* Pass eigenvectors to k_means and perform k_means in K-dimension */
    for (int i = 0; i < data_num; i++)
    {
        U[i] = vector<double>(num_of_cl);
        for (int j = 0; j < num_of_cl; j++)
        {
            U[i][j] = u(i, j);
            output_file << u(i, j) << ",";
        }
        output_file << "\n";
    }
    output_file.close();
    cout << "finished\n";

    // Retrieve results

    k_means(U, data_num, num_of_cl, kernel_choose, 0, gamma);

    U.clear();

    /* normalized cut  D-1 L*/
    cout << "Normalized Cut\n";
    ofstream output_file_normal("normal_cut_data_point_k_" + to_string(kernel_choose) + "_d_" + to_string(data_num) + "_g_" + to_string(g) + ".csv", ios::app);

    for (int i = 0; i < data_num; i++)
    {
        for (int j = 0; j < data_num; j++)
        {
            L(i, j) = L(i, j) / D[i];
        }
    }
    DenseSymMatProd<double> op_normal(L);
    SymEigsSolver<double, SMALLEST_ALGE, DenseSymMatProd<double>> eigs_normal(&op_normal, 5 + 1, 2 * (5 + 1 + 1));
    eigs_normal.init();
    cout << "compute eigenvectors\n";
    eigs_normal.compute();
    cout << "finished\n";
    Eigen::MatrixXd u_normal;
    vector<vector<double>> U_normal(data_num);
    if (eigs_normal.info() == SUCCESSFUL)
    {
        u_normal = eigs_normal.eigenvectors();
    }
    else
    {
        cout << "fail to compute eigenvectors\n";
        return -1;
    }

    cout << "Put eigenvectors in U_normal\n";
    for (int i = 0; i < data_num; i++)
    {
        U_normal[i] = vector<double>(num_of_cl);
        for (int j = 0; j < num_of_cl; j++)
        {
            U_normal[i][j] = u_normal(i, j);
            output_file_normal << u_normal(i, j) << ",";
        }
        output_file_normal << "\n";
    }
    output_file_normal.close();
    cout << "finished\n";

    // Retrieve results

    k_means(U_normal, data_num, num_of_cl, kernel_choose, 1, gamma);

    return 0;
};

int LDA(int data_num, int num_of_cl)
{
    cout << "LDA\n";
    if (data_num == 5000)
    {
        vector<vector<double>> centers(num_of_cl);
        vector<int> cl_elements(num_of_cl);

        /* Retrieve ground truth labels */
        ifstream input_file("T_train.csv");
        string num;
        if (!input_file.is_open())
            return -1;
        cout << "Read target file\n";
        for (int i = 0; i < data_num; i++)
        {
            getline(input_file, num, '\n');
            T_train[i] = stod(num);
        }

        cout << "end\nFind centers\n";

        /* initialize centers */
        for (int i = 0; i < num_of_cl; i++)
        {
            cl_elements[i] = 0;
            centers[i] = vector<double>(784);
            for (int j = 0; j < 784; j++)
            {
                centers[i][j] = 0;
            }
        }

        /* Find centers for each class */
        for (int i = 0; i < data_num; i++)
        {
            cl_elements[T_train[i] - 1]++;

            for (int j = 0; j < 784; j++)
            {
                centers[T_train[i] - 1][j] += X_train[i][j];
            }
        }
        for (int i = 0; i < num_of_cl; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                centers[i][j] /= cl_elements[i];
            }
        }

        cout << "Finish centers computation\n";

        double tmp_x[784] = {0};
        double tmp_sb_x[784] = {0};
        Eigen::MatrixXd Sw(784, 784);
        Eigen::MatrixXd Sb(784, 784);
        /* initialize Sw and Sb */
        for (int j = 0; j < 784; j++)
        {
            for (int l = 0; l < 784; l++)
            {
                Sb(j, l) = 0;
                Sw(j, l) = 0;
            }
        }

        /* compute Sw */
        for (int i = 0; i < data_num; i++)
        {
            cout << "Sw data " << i << endl;

            for (int j = 0; j < 784; j++)
            {
                tmp_x[j] = X_train[i][j] - centers[T_train[i] - 1][j];
            }

            for (int j = 0; j < 784; j++)
            {
                for (int l = 0; l < 784; l++)
                {
                    Sw(j, l) += tmp_x[j] * tmp_x[l];
                }
            }
        }

        /* compute Sb */
        for (int k = 0; k < num_of_cl; k++)
        {
            cout << "Sb Cluster " << k << endl;
            for (int j = 0; j < 784; j++)
            {
                tmp_sb_x[j] = centers[k][j] - X_train_c[j];
            }

            for (int j = 0; j < 784; j++)
            {
                for (int l = 0; l < 784; l++)
                {
                    Sb(j, l) += tmp_sb_x[j] * tmp_sb_x[l] * cl_elements[k];
                }
            }
        }

        cout << "Sw_inv_Sb compute" << endl;
        Eigen::MatrixXd Sw_inv_Sb = Sw.completeOrthogonalDecomposition().pseudoInverse() * Sb;
        DenseSymMatProd<double> op(Sw_inv_Sb);
        SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, 2, 2 * (2 + 1));
        eigs.init();
        eigs.compute();

        cout << "Sw_inv_Sb computation finished" << endl;
        Eigen::MatrixXd evectors;
        if (eigs.info() == SUCCESSFUL)
        {
            evectors = eigs.eigenvectors();
        }

        ofstream ouput_train_2D("LDA_train.txt", ios::app);
        double lda_2d[2] = {0};

        for (int i = 0; i < data_num; i++)
        {
            cout << "data " << i << endl;
            for (int j = 0; j < 784; j++)
            {
                lda_2d[0] += X_train[i][j] * evectors(j, 0);
                lda_2d[1] += X_train[i][j] * evectors(j, 1);
            }
            ouput_train_2D << lda_2d[0] << "," << lda_2d[1] << "\n";
            lda_2d[0] = 0;
            lda_2d[1] = 0;
        }
        ouput_train_2D.close();

        ofstream output_centers("LDA_train_centers.txt", ios::app);
        for (int i = 0; i < num_of_cl; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                lda_2d[0] += centers[i][j] * evectors(j, 0);
                lda_2d[1] += centers[i][j] * evectors(j, 1);
            }
            output_centers << lda_2d[0] << "," << lda_2d[1] << "\n";
            lda_2d[0] = 0;
            lda_2d[1] = 0;
        }
        for (int j = 0; j < 784; j++)
        {
            lda_2d[0] += X_train_c[j] * evectors(j, 0);
            lda_2d[1] += X_train_c[j] * evectors(j, 1);
        }
        output_centers << lda_2d[0] << "," << lda_2d[1] << "\n";
        output_centers.close();
    }
    else
    {
        vector<vector<double>> centers(num_of_cl);
        vector<int> cl_elements(num_of_cl);

        /* Retrieve ground truth labels */
        ifstream input_file("T_test.csv");
        string num;
        if (!input_file.is_open())
            return -1;
        for (int i = 0; i < data_num; i++)
        {
            getline(input_file, num, '\n');
            T_test[i] = stod(num);
        }

        cout << "end\nFind centers\n";

        /* initialize centers */
        for (int i = 0; i < num_of_cl; i++)
        {
            cl_elements[i] = 0;
            centers[i] = vector<double>(784);
            for (int j = 0; j < 784; j++)
            {
                centers[i][j] = 0;
            }
        }

        /* Find centers */
        for (int i = 0; i < data_num; i++)
        {
            cout << "data " << i << endl;
            cl_elements[T_test[i] - 1]++;

            for (int j = 0; j < 784; j++)
            {
                centers[T_test[i] - 1][j] += X_test[i][j];
            }
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                centers[i][j] /= cl_elements[i];
            }
        }

        cout << "Finish centers computation\n";

        double tmp_x[784] = {0};
        double tmp_sb_x[784] = {0};
        Eigen::MatrixXd Sw(784, 784);
        Eigen::MatrixXd Sb(784, 784);

        /* initialize Sw and Sb */
        for (int j = 0; j < 784; j++)
        {
            for (int l = 0; l < 784; l++)
            {
                Sb(j, l) = 0;
                Sw(j, l) = 0;
            }
        }

        /* compute Sw */
        for (int i = 0; i < data_num; i++)
        {
            cout << "Sw data " << i << endl;

            for (int j = 0; j < 784; j++)
            {
                tmp_x[j] = X_test[i][j] - centers[T_test[i] - 1][j];
            }

            for (int j = 0; j < 784; j++)
            {
                for (int l = 0; l < 784; l++)
                {
                    Sw(j, l) += tmp_x[j] * tmp_x[l];
                }
            }
        }

        /* compute Sb */
        for (int k = 0; k < num_of_cl; k++)
        {
            cout << "Sb Cluster " << k << endl;
            for (int j = 0; j < 784; j++)
            {
                tmp_sb_x[j] = centers[k][j] - X_test_c[j];
            }

            for (int j = 0; j < 784; j++)
            {
                for (int l = 0; l < 784; l++)
                {
                    Sb(j, l) += tmp_sb_x[j] * tmp_sb_x[l] * cl_elements[k];
                }
            }
        }
        cout << "Sw_inv_Sb compute" << endl;
        Eigen::MatrixXd Sw_inv_Sb = Sw.completeOrthogonalDecomposition().pseudoInverse() * Sb;
        DenseSymMatProd<double> op(Sw_inv_Sb);
        SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, 2, 2 * (2 + 1));
        eigs.init();
        eigs.compute();

        cout << "Sw_inv_Sb computation finished" << endl;
        Eigen::MatrixXd evectors;
        if (eigs.info() == SUCCESSFUL)
        {
            evectors = eigs.eigenvectors();
        }

        ofstream ouput_test_2D("LDA_test.txt", ios::app);
        double lda_2d[2] = {0};

        for (int i = 0; i < data_num; i++)
        {
            cout << "data " << i << endl;
            for (int j = 0; j < 784; j++)
            {
                lda_2d[0] += X_test[i][j] * evectors(j, 0);
                lda_2d[1] += X_test[i][j] * evectors(j, 1);
            }
            ouput_test_2D << lda_2d[0] << "," << lda_2d[1] << "\n";
            lda_2d[0] = 0;
            lda_2d[1] = 0;
        }
        ouput_test_2D.close();

        ofstream output_centers("LDA_test_centers.txt", ios::app);
        for (int i = 0; i < num_of_cl; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                lda_2d[0] += centers[i][j] * evectors(j, 0);
                lda_2d[1] += centers[i][j] * evectors(j, 1);
            }
            output_centers << lda_2d[0] << "," << lda_2d[1] << "\n";
            lda_2d[0] = 0;
            lda_2d[1] = 0;
        }
        for (int j = 0; j < 784; j++)
        {
            lda_2d[0] += X_test_c[j] * evectors(j, 0);
            lda_2d[1] += X_test_c[j] * evectors(j, 1);
        }
        output_centers << lda_2d[0] << "," << lda_2d[1] << "\n";
        output_centers.close();
    }
    return 0;
};

int att_faces()
{
    cout << "PCA face part\n";
    string line;
    int len, wid, iter = 0, im_num = 400;
    unsigned char pixel;
    unsigned int contain;
    double pix[92 * 112];
    double center[92 * 112] = {0};

    /* read file to compute the center */
    cout << "compute center\n";
    for (int j = 1; j <= 10; j++)
    {
        for (int i = 1; i <= 40; i++)
        {
            ifstream input_file("att_faces/s" + to_string(i) + "/" + to_string(j) + ".pgm", ios::binary);
            stringstream ss;
            if (input_file.is_open())
            {
                getline(input_file, line, '\n');
                getline(input_file, line, ' ');
                len = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                wid = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                cout << i << " The picture is " << len << " x " << wid << endl;
                ss << input_file.rdbuf();
                iter = 0;
                for (int k = 0; k < len; k++)
                {
                    for (int l = 0; l < wid; l++)
                    {

                        ss >> pixel;
                        //getline(input_file, line, '\n');
                        contain = pixel;
                        pix[iter] = (double)contain;
                        center[iter] += pix[iter];
                        iter++;
                    }
                }
            }
        }
    }

    for (int j = 0; j < 92 * 112; j++)
    {
        center[j] /= im_num;
    }
    cout << "Finish\n";

    /* compute S */
    Eigen::MatrixXd S(92 * 112, 92 * 112);
    cout << "Produce S" << endl;
    for (int j = 1; j <= 10; j++)
    {
        for (int i = 1; i <= 40; i++)
        {
            ifstream input_file("att_faces/s" + to_string(i) + "/" + to_string(j) + ".pgm", ios::binary);
            stringstream ss;
            if (input_file.is_open())
            {
                getline(input_file, line, '\n');
                getline(input_file, line, ' ');
                len = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                wid = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                cout << i << " " << j << " The picture is " << len << " x " << wid << endl;
                ss << input_file.rdbuf();
                iter = 0;
                /* compute S */
                for (int k = 0; k < len; k++)
                {
                    for (int l = 0; l < wid; l++)
                    {

                        ss >> pixel;
                        contain = pixel;
                        pix[iter] = (double)contain;
                        pix[iter] = pix[iter] - center[iter];
                        iter++;
                    }
                }

                for (int l = 0; l < 92 * 112; l++)
                {
                    for (int k = 0; k < 92 * 112; k++)
                    {
                        S(l, k) += pix[l] * pix[k] / im_num;
                    }
                }
            }
        }
    }

    cout << "Start compute eigenvectors\n";
    DenseSymMatProd<double> op(S);
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, 40, 2 * (40 + 1));
    eigs.init();
    eigs.compute();
    Eigen::MatrixXd evectors;
    if (eigs.info() == SUCCESSFUL)
    {
        evectors = eigs.eigenvectors();
    }
    cout << "end\n";
    Eigen::MatrixXd WWt = evectors * evectors.transpose();
    cout << "Compute transpose\nStart to write new eigenface in\n";

    double bitmap[112][92];
    for (int i = 0; i < 40; i++)
    {
        cout << "Write ..." << i << endl;
        ofstream output_2D("new_eigen_face_" + to_string(i) + ".txt", ios::app);
        /* eigen face */
        iter = 0;
        output_2D << i << '\n';
        for (int j = 0; j < 112; j++)
        {
            for (int k = 0; k < 92 - 1; k++)
            {
                bitmap[j][k] = evectors(iter, i);
                iter++;
                output_2D << bitmap[j][k] << ",";
            }

            output_2D << bitmap[j][92 - 1] << '\n';
        }

        output_2D.close();
    }

    /* Reconstruct original Face */
    for (int j = 1; j <= 1; j++)
    {
        for (int i = 1; i <= 10; i++)
        {
            ifstream input_file("att_faces/s" + to_string(i) + "/" + to_string(j) + ".pgm", ios::binary);
            stringstream ss;
            if (input_file.is_open())
            {
                getline(input_file, line, '\n');
                getline(input_file, line, ' ');
                len = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                wid = stoi(line, nullptr, 0);
                getline(input_file, line, '\n');
                cout << i << " The picture is " << len << " x " << wid << endl;
                ss << input_file.rdbuf();
                iter = 0;
                for (int k = 0; k < len; k++)
                {
                    for (int l = 0; l < wid; l++)
                    {

                        ss >> pixel;
                        contain = pixel;
                        pix[iter] = (double)contain;
                        iter++;
                    }
                }
                /* finish reading a picture */

                ofstream output_2D_data("new_recon_face_" + to_string(i) + "_" + to_string(j) + ".txt", ios::app);

                /* Reconstruction */
                for (int l = 0; l < 112; l++)
                {
                    for (int k = 0; k < 92; k++)
                    {
                        bitmap[l][k] = 0;
                    }
                }

                for (int l = 0; l < 92 * 112; l++)
                {
                    for (int k = 0; k < 92 * 112; k++)
                    {
                        bitmap[l / 92][l % 92] += pix[k] * WWt(k, l);
                    }
                }

                for (int l = 0; l < 112; l++)
                {
                    for (int k = 0; k < 92 - 1; k++)
                    {
                        output_2D_data << bitmap[l][k] << ",";
                    }
                    output_2D_data << bitmap[l][92 - 1] << '\n';
                }
                output_2D_data.close();
            }
        }
    }

    return 0;
};

int main()
{
    srand(time(NULL));
    int write_file, kernel_choose, data_mode, lda_mode, face_mode;
    double g1, g2, g_step;
    reader(0);

    /* LDA part */
    cout << "Do LDA?\n";
    cin >> lda_mode;
    if (lda_mode == 1)
    {
        LDA(5000, 5);
        LDA(2500, 5);
        return 0;
    }
    cout << "Do pca for att_faces?\n";
    cin >> face_mode;
    if (face_mode == 1)
    {
        att_faces();
        return 0;
    }

    cout << "write pca results to files?\n";
    cin >> write_file;

    cout << "end of PCA\n>> Kernel_choose\n";
    cin >> kernel_choose;
    cout << ">> data_mode\n"; // 0 is train
    cin >> data_mode;
    if (kernel_choose != 2)
    {
        cout << "Gamma range and step\n";
        // cin >> g1 >> g2 >> g_step;
        PCA(1, 1, 5000, 0.01);
        Spectral(1, 5, 5000, 0.01);
        PCA(1, 0, 5000, 1);
        Spectral(0, 5, 5000, 1);
        PCA(1, 1, 5000, 20);
        Spectral(1, 5, 5000, 20);
        PCA(1, 0, 5000, 30);
        Spectral(0, 5, 5000, 30);
        PCA(1, 1, 5000, 90);
        Spectral(1, 5, 5000, 90);
        PCA(1, 0, 5000, 60);
        Spectral(0, 5, 5000, 60);
        PCA(1, 0, 5000, 90);
        Spectral(0, 5, 5000, 90);

        // Spectral(2, 5, 5000, 0.0);
        // Spectral(2, 5, 2500, 0.0);
    }
    else
    { // use linear
        if (data_mode == 0)
        {
            Spectral(kernel_choose, 5, 5000, 0.0);
        }
        else
        {
            Spectral(kernel_choose, 5, 2500, 0.0);
        }
    }

    return 0;
}