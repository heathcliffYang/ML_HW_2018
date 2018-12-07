#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <SymEigsSolver.h>

using namespace Spectra;
using namespace std;

typedef struct data_point
{
    double x;
    double y;
    int cluster;
} data_point;

typedef struct cluster_center
{
    double x;
    double y;
    int num_of_element;
} cluster_center;

/* data points record */
vector<data_point> data_points;

vector<cluster_center> centers;

int reader()
{
    string line;
    ifstream input_file("circle.txt");

    data_points.clear();

    if (input_file.is_open())
    {
        while (getline(input_file, line, ','))
        {
            data_point input_num;
            input_num.x = stod(line);
            getline(input_file, line);
            input_num.y = stod(line);
            input_num.cluster = -1;

            //cout << input_num.x << " " << input_num.y << endl;
            data_points.push_back(input_num);
        }
        input_file.close();
        cout << "The # of data is " << data_points.size() << endl;
    }
    else
    {
        cout << "Cannot open" << endl;
        return -1;
    }

    return 0;
};

// double compute_Eu_distance(data_point a, cluster_center c)
// {
//     return ;
// };

cluster_center old;

bool is_converge()
{
    /* Check out if centers become steady */
    static vector<cluster_center> old_centers(centers.size(), old);
    double error = 0;

    for (int i = 0; i < centers.size(); i++)
    {
        cout << "converge check :" << old_centers[i].x << " " << centers[i].x << endl;
        error += sqrt(pow(old_centers[i].x - centers[i].x, 2) + pow(old_centers[i].y - centers[i].y, 2));
        old_centers[i].y = centers[i].y;
        old_centers[i].x = centers[i].x;
    }

    if (error / centers.size() < 0.00001)
    {
        cout << "converge!\n";
        return false;
    }

    return true;
};

int init_centers(int num_of_cl, int init_mode)
{
    switch (init_mode)
    {
    case 1:
        cout << "CCIA" << endl;
        break;
    case 2:
        cout << "" << endl;
        break;
    default:
        cout << "Random" << endl;
        for (int i = 0; i < num_of_cl; i++)
        {
            cluster_center center;
            center.x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            center.y = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            center.num_of_element = i;
            centers.push_back(center);
        }
    }

    return 0;
};

int K_Means(int num_of_cl, int init_mode)
{
    centers.clear();

    init_centers(num_of_cl, init_mode);

    double min = 0, tmp = 0;
    int belong_index = 0, iter = 0;
    min = 100;
    ofstream output_file("k-means_cl_results.txt", ios::app);
    do
    {
        iter++;
        cout << "iter " << iter << endl;
        /* E-step: find the nearest center */
        for (int i = 0; i < data_points.size(); i++)
        {
            for (int j = 0; j < centers.size(); j++)
            {
                tmp = sqrt(pow(data_points[i].x - centers[j].x, 2) + pow(data_points[i].y - centers[j].y, 2));
                if (min > tmp)
                {
                    min = tmp;
                    belong_index = j;
                }
            }
            centers[belong_index].num_of_element++;
            data_points[i].cluster = belong_index;
            min = 100;
        }

        /* M-step: update centers */
        for (int i = 0; i < data_points.size(); i++)
        {
            centers[data_points[i].cluster].x += data_points[i].x;
            centers[data_points[i].cluster].y += data_points[i].y;
            /* write the new clustering */
            output_file << data_points[i].cluster << '\n';
        }

        output_file << -1 << '\n';

        for (int j = 0; j < centers.size(); j++)
        {
            centers[j].x /= centers[j].num_of_element;
            centers[j].y /= centers[j].num_of_element;
            centers[j].num_of_element = 0;
        }

    } while (is_converge());

    output_file.close();
    return 0;
};

double RBF(int a, int b, double gamma)
{
    return exp(-gamma * pow(data_points[a].x - data_points[b].x, 2) + pow(data_points[a].y - data_points[b].y, 2));
};

bool kernel_is_converge()
{
    /* Check out if centers become steady */
    static vector<cluster_center> old_centers(centers.size(), old);
    int error = 0;

    for (int i = 0; i < centers.size(); i++)
    {
        cout << "converge check :" << old_centers[i].num_of_element << " " << centers[i].num_of_element << endl;
        error += abs(old_centers[i].num_of_element - centers[i].num_of_element);
        old_centers[i].num_of_element = centers[i].num_of_element;
    }

    if (error == 0)
    {
        cout << "converge!\n";
        return false;
    }

    return true;
};

int Kernel_k_means(int num_of_cl, double gamma)
{
    vector<double> tmp(num_of_cl, 0.0);
    vector<double> cluster_base(num_of_cl, 0.0);
    vector<double>::iterator min;
    int belong_index = 0, iter = 0;
    string file_name = "kernel-k-means_" + to_string(int(gamma));
    ofstream output_file(file_name, ios::app);

    /* initial */
    centers.clear();
    for (int i = 0; i < num_of_cl; i++)
    {
        cluster_center center;
        center.num_of_element = 0;
        centers.push_back(center);
    }
    int ind = -1;
    for (int i = 0; i < data_points.size(); i++)
    {
        ind = rand() % num_of_cl;
        centers[ind].num_of_element++;
        data_points[i].cluster = ind;
        /* write the new clustering */
        output_file << ind << '\n';
    }

    output_file << -1;

    for (int i = 0; i < num_of_cl; i++)
    {
        output_file << "," << centers[i].num_of_element;
    }

    output_file << '\n';

    do
    {
        iter++;
        cout << "iter " << iter << endl;

        for (int i = 0; i < data_points.size(); i++)
        {
            for (int j = 0; j < data_points.size(); j++)
            {
                if (data_points[i].cluster == data_points[j].cluster)
                    cluster_base[data_points[i].cluster] += RBF(i, j, gamma);
                //cout << i << "     " << j << "     " << cluster_base[data_points[i].cluster] << endl;
            }
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            if (centers[i].num_of_element != 0)
                cluster_base[i] /= pow(centers[i].num_of_element, 2);
            else
                cluster_base[i] = 0;
            cout << "cluster base " << i << " contains " << centers[i].num_of_element << " is " << cluster_base[i] << endl;
        }

        for (int i = 0; i < data_points.size(); i++)
        {
            for (int j = 0; j < data_points.size(); j++)
            {
                tmp[data_points[j].cluster] -= RBF(i, j, gamma);
            }

            for (int k = 0; k < num_of_cl; k++)
            {
                if (centers[k].num_of_element != 0)
                {
                    tmp[k] /= centers[k].num_of_element;
                    tmp[k] *= 2;
                    cout << "data " << i << " simialr second is " << tmp[k] << '\b';
                    tmp[k] += cluster_base[k];
                }
                cout << "data " << i << " simialr with " << k << " is " << tmp[k] << endl;
            }

            centers[data_points[i].cluster].num_of_element--;

            min = max_element(tmp.begin(), tmp.end());
            data_points[i].cluster = distance(tmp.begin(), min);
            centers[data_points[i].cluster].num_of_element++;
            output_file << data_points[i].cluster << '\n';

            tmp.assign(num_of_cl, 0);
        }

        cluster_base.assign(num_of_cl, 0);

        output_file << -1;

        for (int i = 0; i < num_of_cl; i++)
        {
            output_file << "," << centers[i].num_of_element;
        }

        output_file << '\n';

        if (iter > 100)
            break;
    } while (kernel_is_converge());

    output_file.close();
    return 0;
};

int Spectral(int num_of_cl, double gamma)
{
    int num_of_data = data_points.size();
    vector<vector<double>> W(num_of_data);

    /* Calculate W */
    for (int i = 0; i < num_of_data; i++)
    {
        for (int j = 0; j < num_of_data; j++)
        {
            W[i][j] = RBF(i, j, gamma);
        }
    }

    /* L */
    Eigen::MatrixXd L(num_of_data, num_of_data);
    for (int i = 0; i < num_of_data; i++)
    {
        L(i, i) = 0;
        for (int j = 0; j < num_of_data; j++)
        {
            if (i != j)
            {
                L(i, j) = -W[i][j];
                L(i, i) += W[i][j];
            }
        }
    }

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSymMatProd<double> op(L);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, num_of_cl, 2 * num_of_cl);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXd evalues;
    if (eigs.info() == SUCCESSFUL)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n"
              << evalues << std::endl;

    //////////////////////////////

    vector<double> cluster_base(num_of_cl, 0.0);
    vector<double>::iterator min;
    int belong_index = 0, iter = 0;
    string file_name = "Spectral_" + to_string(int(gamma));
    ofstream output_file(file_name, ios::app);

    /* initial */
    centers.clear();
    for (int i = 0; i < num_of_cl; i++)
    {
        cluster_center center;
        center.num_of_element = 0;
        centers.push_back(center);
    }
    int ind = -1;
    for (int i = 0; i < data_points.size(); i++)
    {
        ind = rand() % num_of_cl;
        centers[ind].num_of_element++;
        data_points[i].cluster = ind;
        /* write the new clustering */
        output_file << ind << '\n';
    }

    output_file << -1;

    for (int i = 0; i < num_of_cl; i++)
    {
        output_file << "," << centers[i].num_of_element;
    }

    output_file << '\n';

    do
    {
        iter++;
        cout << "iter " << iter << endl;

        for (int i = 0; i < data_points.size(); i++)
        {
            for (int j = 0; j < data_points.size(); j++)
            {
                if (data_points[i].cluster == data_points[j].cluster)
                    cluster_base[data_points[i].cluster] += RBF(i, j, gamma);
                //cout << i << "     " << j << "     " << cluster_base[data_points[i].cluster] << endl;
            }
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            if (centers[i].num_of_element != 0)
                cluster_base[i] /= pow(centers[i].num_of_element, 2);
            else
                cluster_base[i] = 0;
            cout << "cluster base " << i << " contains " << centers[i].num_of_element << " is " << cluster_base[i] << endl;
        }

        for (int i = 0; i < data_points.size(); i++)
        {
            for (int j = 0; j < data_points.size(); j++)
            {
                tmp[data_points[j].cluster] -= RBF(i, j, gamma);
            }

            for (int k = 0; k < num_of_cl; k++)
            {
                if (centers[k].num_of_element != 0)
                {
                    tmp[k] /= centers[k].num_of_element;
                    tmp[k] *= 2;
                    cout << "data " << i << " simialr second is " << tmp[k] << '\b';
                    tmp[k] += cluster_base[k];
                }
                cout << "data " << i << " simialr with " << k << " is " << tmp[k] << endl;
            }

            centers[data_points[i].cluster].num_of_element--;

            min = max_element(tmp.begin(), tmp.end());
            data_points[i].cluster = distance(tmp.begin(), min);
            centers[data_points[i].cluster].num_of_element++;
            output_file << data_points[i].cluster << '\n';

            tmp.assign(num_of_cl, 0);
        }

        cluster_base.assign(num_of_cl, 0);

        output_file << -1;

        for (int i = 0; i < num_of_cl; i++)
        {
            output_file << "," << centers[i].num_of_element;
        }

        output_file << '\n';

        if (iter > 100)
            break;
    } while (kernel_is_converge());

    output_file.close();
    return 0;
};

int main()
{
    srand(time(NULL));
    reader();
    cout << "K-means (1), kernel k-means (2), spectral clustering (3):" << endl;
    int clustering_mode, num_of_cl, init_mode;
    cin >> clustering_mode;
    double n1, n2;

    while (true)
    {
        switch (clustering_mode)
        {
        case 1:
            cout << "K-means\nThe # of cluseter you want: ";
            cin >> num_of_cl;
            cout << "The initialization methods, CCIA (1), another (2), random (others): ";
            cin >> init_mode;
            K_Means(num_of_cl, init_mode);
            clustering_mode = 4;
            break;
        case 2:
            cout << "Kernel k-means\nThe # of cluseter you want: ";
            cin >> num_of_cl;
            cout << "The gamma search range is from n1 to n2: ";
            cin >> n1 >> n2;
            for (double gamma = n1; gamma < n2; gamma += 25.0)
            {
                Kernel_k_means(num_of_cl, gamma);
            }
            clustering_mode = 4;
            break;
        case 3:
            cout << "Spectral clustering\n";
            clustering_mode = 4;
            break;
        case 4:
            cout << "K-means (1), kernel k-means (2), spectral clustering (3):" << endl;
            cin >> clustering_mode;
            break;
        default:
            cout << "End\n";
            data_points.~vector();
            centers.~vector();
            clustering_mode = 0;
        }
        if (clustering_mode > 4 || clustering_mode < 1)
            break;
    }

    return 0;
}