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

vector<double> cluster_base;

int reader(int data_mode)
{
    string line;
    string file_name;
    if (data_mode == 0)
        file_name = "circle.txt";
    else
        file_name = "moon.txt";
    ifstream input_file(file_name);

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
    {
        cout << "CCIA" << endl;
        vector<double> distribution_x(data_points.size());
        vector<int> tag_x(data_points.size());

        for (int i = 0; i < data_points.size(); i++)
        {
            distribution_x[i] = data_points[i].x;
        }
        sort(distribution_x.begin(), distribution_x.end());
        for (int i = 0; i < data_points.size(); i++)
        {
            if (data_points[i].x < distribution_x[floor(data_points.size() / 2)])
                tag_x[i] = 0;
            else
                tag_x[i] = 1;
        }
        vector<double> distribution_y(data_points.size());
        vector<int> tag_y(data_points.size());

        for (int i = 0; i < data_points.size(); i++)
        {
            distribution_y[i] = data_points[i].x;
        }
        sort(distribution_y.begin(), distribution_y.end());

        int pattern[2][2] = {};

        for (int i = 0; i < data_points.size(); i++)
        {

            if (data_points[i].x < distribution_y[floor(data_points.size() / 2)])
            {
                tag_y[i] = 0;
                pattern[tag_x[i]][tag_y[i]]++;
            }
            else
            {
                tag_y[i] = 1;
                pattern[tag_x[i]][tag_y[i]]++;
            }
        }
        ofstream output_file("k-means_cl_ccia.txt", ios::app);
        output_file << -1;

        int max_x[2] = {}, max_y[2] = {}, max[2] = {};
        for (int k = 0; k < 2; k++)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (pattern[i][j] > max[k])
                    {
                        max[k] = pattern[i][j];
                        max_x[k] = i;
                        max_y[k] = j;
                    }
                }
            }
            pattern[max_x[k]][max_y[k]] = 0;
            cluster_center center;
            if (max_x[k] == 0)
                center.x = distribution_x[int(data_points.size() / 3)];
            else
                center.x = distribution_x[int(data_points.size() * 2 / 3)];
            output_file << ',' << center.x;
            if (max_y[k] == 0)
                center.y = distribution_y[int(data_points.size() / 3)];
            else
                center.y = distribution_y[int(data_points.size() * 2 / 3)];
            output_file << ',' << center.y;
            center.num_of_element = 0;
            centers.push_back(center);
        }
        output_file << '\n';
        output_file.close();
        break;
    }
    default:
        cout << "Random" << endl;
        ofstream output_file("k-means_cl_random.txt", ios::app);
        output_file << -1;
        for (int i = 0; i < num_of_cl; i++)
        {
            cluster_center center;
            center.x = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            center.y = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            output_file << ',' << center.x;
            output_file << ',' << center.y;
            center.num_of_element = 0;
            centers.push_back(center);
        }
        output_file << '\n';
        output_file.close();
    }
    cout << "End\n"
         << endl;
    return 0;
};

int K_Means(int num_of_cl, int init_mode, int write_file)
{
    centers.clear();
    init_centers(num_of_cl, init_mode);
    double min = 0, tmp = 0;
    int belong_index = 0, iter = 0;
    min = 100;
    ofstream output_file("k-means_cl_ccia.txt", ios::app);
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
        for (int j = 0; j < centers.size(); j++)
        {
            centers[j].x = 0;
            centers[j].y = 0;
        }
        for (int i = 0; i < data_points.size(); i++)
        {
            centers[data_points[i].cluster].x += data_points[i].x;
            centers[data_points[i].cluster].y += data_points[i].y;
            /* write the new clustering */
            if (write_file == 1)
                output_file << data_points[i].cluster << '\n';
        }
        if (write_file == 1)
            output_file << -1;

        for (int j = 0; j < centers.size(); j++)
        {
            centers[j].x /= centers[j].num_of_element;
            centers[j].y /= centers[j].num_of_element;
            if (write_file == 1)
            {
                output_file << ',' << centers[j].x << ',' << centers[j].y;
                centers[j].num_of_element = 0;
            }
        }
        if (write_file == 1)
            output_file << '\n';

    } while (is_converge() && write_file == 1);

    output_file.close();
    return 0;
};

double RBF(int a, int b, double gamma)
{
    return exp(-gamma * (pow(data_points[a].x - data_points[b].x, 2) + pow(data_points[a].y - data_points[b].y, 2)));
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

int Kernel_k_means(int num_of_cl, double gamma, int init_mode)
{
    vector<double> tmp(num_of_cl, 0.0);
    cluster_base.assign(num_of_cl, 0.0);
    vector<int> next_distribution(data_points.size(), 0);
    vector<double>::iterator min;
    int belong_index = 0, iter = 0;
    string file_name;

    /* initial */
    switch (init_mode)
    {
    case 1:
    {
        cout << "CCIA" << endl;
        file_name = "kernel-k-means_ccia_c" + to_string(num_of_cl) + "_g" + to_string(int(gamma));
        K_Means(num_of_cl, init_mode, 0);
        break;
    }
    default:
        cout << "Random" << endl;
        file_name = "kernel-k-means_random_c" + to_string(num_of_cl) + "_g" + to_string(int(gamma));
        ofstream output_file(file_name, ios::app);
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
        output_file.close();
    }

    ofstream output_file(file_name, ios::app);

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
            cout << "cluster base " << cluster_base[i] << " ";
            if (centers[i].num_of_element != 0)
                cluster_base[i] /= pow(centers[i].num_of_element, 2);
            else
                cluster_base[i] = 0;
            cout << i << " contains " << centers[i].num_of_element << " is " << cluster_base[i] << endl;
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
                    cout << "data " << i << " simialr second is " << tmp[k] << ' ';
                    tmp[k] += cluster_base[k];
                }
                cout << "data " << i << " simialr with " << k << " is " << tmp[k] << endl;
            }

            //centers[data_points[i].cluster].num_of_element--;

            min = min_element(tmp.begin(), tmp.end());
            //data_points[i].cluster = distance(tmp.begin(), min);
            next_distribution[i] = distance(tmp.begin(), min);
            cout << next_distribution[i] << endl;
            //centers[data_points[i].cluster].num_of_element++;
            output_file << next_distribution[i] << '\n';

            tmp.assign(num_of_cl, 0);
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            centers[i].num_of_element = 0;
        }

        for (int i = 0; i < data_points.size(); i++)
        {
            centers[next_distribution[i]].num_of_element++;
            data_points[i].cluster = next_distribution[i];
        }

        cluster_base.assign(num_of_cl, 0);

        output_file << -1;

        for (int i = 0; i < num_of_cl; i++)
        {
            output_file << "," << centers[i].num_of_element;
        }

        output_file << '\n';

        // if (iter > 3)
        //     break;
    } while (kernel_is_converge());

    output_file.close();
    return 0;
};

int Spectral(int num_of_cl, double gamma, int init_mode)
{
    int num_of_data = data_points.size();
    cout << "# of data " << num_of_data << endl;
    vector<vector<double>> W(num_of_data);

    /* Calculate W */
    for (int i = 0; i < num_of_data; i++)
    {
        data_points[i].cluster = 0;
        W[i] = vector<double>(num_of_data);
        for (int j = 0; j < num_of_data; j++)
        {
            W[i][j] = RBF(i, j, gamma);
        }
    }
    cout << "L start\n";
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

    cout << "L end\n";

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSymMatProd<double> op(L);
    cout << "1\n";
    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver<double, SMALLEST_ALGE, DenseSymMatProd<double>> eigs(&op, num_of_cl + 1, 2 * (num_of_cl + 1));
    cout << "2\n";
    // Initialize and compute
    eigs.init();
    cout << "3\n";
    int nconv = eigs.compute();
    cout << "4\n";
    // Retrieve results
    Eigen::VectorXd evalues;
    Eigen::MatrixXd evectors;
    if (eigs.info() == SUCCESSFUL)
    {
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
        cout << "5\n";
    }
    cout << "get eigen" << endl;

    string file_name = "Spectral_eigenvector_c" + to_string(num_of_cl) + "_g" + to_string(int(gamma));
    ofstream output_file(file_name, ios::app);

    for (int j = 0; j < evectors.cols(); j++)
    {
        for (int i = 0; i < evectors.rows(); i++)
        {
            output_file << evectors(i, j) << '\n';
        }
        output_file << -1 << '\n';
    }
    output_file.close();

    string distri_file_name;
    cout << "Eigen space K-means!" << endl;

    /* k-means init  remember centers!!! */
    switch (init_mode)
    {
    case 1:
    {
        init_centers(num_of_cl, init_mode);
        distri_file_name = "Spectral_ccia_c" + to_string(num_of_cl) + "_g" + to_string(int(gamma));
    }
    default:
        distri_file_name = "Spectral_random_c" + to_string(num_of_cl) + "_g" + to_string(int(gamma));
        init_centers(num_of_cl, init_mode);
    }
    ofstream distri_output_file(distri_file_name, ios::app);
    int itr = 0;
    do
    {
        itr++;
        /* E-step */
        double tmp = 0, min = MAXFLOAT;
        for (int i = 0; i < evectors.rows(); i++)
        {

            for (int k = 0; k < num_of_cl; k++)
            {
                tmp = sqrt(pow(evectors(i, 0) - centers[k].x, 2) + pow(evectors(i, 1) - centers[k].y, 2));

                if (min > tmp)
                {
                    min = tmp;
                    data_points[i].cluster = k;
                }
            }
            min = MAXFLOAT;
            centers[data_points[i].cluster].num_of_element++;
            distri_output_file << data_points[i].cluster << '\n';
        }

        distri_output_file << -1 << '\n';

        /* M-step */
        for (int i = 0; i < num_of_cl; i++)
        {
            centers[i].x = 0;
            centers[i].y = 0;
        }

        for (int i = 0; i < data_points.size(); i++)
        {
            centers[data_points[i].cluster].x += evectors(i, 0);
            centers[data_points[i].cluster].y += evectors(i, 1);
        }

        for (int i = 0; i < num_of_cl; i++)
        {
            if (centers[i].num_of_element != 0)
            {
                centers[i].x /= centers[i].num_of_element;
                centers[i].y /= centers[i].num_of_element;
            }
        }
        for (int i = 0; i < num_of_cl; i++)
        {
            centers[i].num_of_element = 0;
        }
        if (itr > 3)
            break;
    } while (is_converge());

    distri_output_file.close();
    return 0;
};

int main()
{
    srand(time(NULL));
    int data_mode;
    cin >> data_mode;
    /* data_mode = 0 means to use circle.txt, 1 is moon.txt */
    reader(data_mode);
    cout << "K-means (1), kernel k-means (2), spectral clustering (3):" << endl;
    int clustering_mode, num_of_cl, init_mode;
    cin >> clustering_mode;
    double n1, n2, step;

    while (true)
    {
        switch (clustering_mode)
        {
        case 1:
            cout << "K-means\nThe # of cluseter you want: \n";
            cin >> num_of_cl;
            cout << "The initialization methods, CCIA (1), random (others): \n";
            cin >> init_mode;
            K_Means(num_of_cl, init_mode, 1);
            clustering_mode = 4;
            break;
        case 2:
            cout << "Kernel k-means\nThe # of cluseter you want: \n";
            cin >> num_of_cl;
            cout << "The initialization methods, CCIA (1), random (others): \n";
            cin >> init_mode;
            cout << "The gamma search range is from n1 to n2 by step: \n";
            cin >> n1 >> n2 >> step;
            for (double gamma = n1; gamma < n2; gamma += step)
            {
                Kernel_k_means(num_of_cl, gamma, init_mode);
            }
            clustering_mode = 4;
            break;
        case 3:
            cout << "Spectral clustering\nThe # of cluseter you want: \n";
            cin >> num_of_cl;
            cout << "The initialization methods, CCIA (1), random (others): \n";
            cin >> init_mode;
            cout << "The gamma search range is from n1 to n2 by step: \n";
            cin >> n1 >> n2 >> step;
            for (double gamma = n1; gamma < n2; gamma += step)
            {
                Spectral(num_of_cl, gamma, init_mode);
            }
            clustering_mode = 4;
            break;
        case 4:
            cout << "K-means (1), kernel k-means (2), spectral clustering (3):" << endl;
            cin >> clustering_mode;
            break;
        default:
            cout << "End\n";
            clustering_mode = 0;
        }
        if (clustering_mode > 4 || clustering_mode < 1)
            break;
    }

    return 0;
}