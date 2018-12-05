#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
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
            getline(input_file, line, ',');
            input_num.y = stod(line);
            input_num.cluster = -1;

            //cout << input_num.x << " " << input_num.y << endl;
            data_points.push_back(input_num);
        }
        input_file.close();
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

int K_Means(int num_of_cl, int init_mode)
{
    centers.clear();
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

    /* E-step: find the nearest center */
    double min = 0, tmp = 0;
    int belong_index = 0, iter = 0;
    min = 100;

    do
    {
        iter++;
        cout << "iter " << iter << endl;
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

        ofstream output_file("k-means_cl_results");
        /* M-step: update centers */
        for (int i = 0; i < data_points.size(); i++)
        {
            centers[data_points[i].cluster].x += data_points[i].x;
            centers[data_points[i].cluster].y += data_points[i].y;
            /* write the new clustering */
            output_file << data_points[i].cluster << '\n';
        }

        for (int j = 0; j < centers.size(); j++)
        {
            centers[j].x /= centers[j].num_of_element;
            centers[j].y /= centers[j].num_of_element;
            centers[j].num_of_element = 0;
        }
    } while (is_converge());

    return 0;
};

int main()
{
    srand(time(NULL));
    reader();
    cout << "K-means (1), kernel k-means (2), spectral clustering (3):" << endl;
    int clustering_mode, num_of_cl, init_mode;
    cin >> clustering_mode;

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
            break;
        case 2:
            cout << "Kernel k-means\n";
            break;
        case 3:
            cout << "Spectral clustering\n";
            break;
        default:
            cout << "End\n";
            data_points.~vector();
            centers.~vector();
        }
        if (clustering_mode > 3 || clustering_mode < 1)
            break;
    }

    // We are going to calculate the eigenvalues of M
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    Eigen::MatrixXd M = A + A.transpose();

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSymMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver<double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXd evalues;
    if (eigs.info() == SUCCESSFUL)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n"
              << evalues << std::endl;

    return 0;
}