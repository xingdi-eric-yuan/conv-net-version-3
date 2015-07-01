#include "general_settings.h"
#include <time.h>
using namespace cv;
using namespace std;

double momentum_w_init = 0.5;
double momentum_d2_init = 0.5;
double momentum_w_adjust = 0.95;
double momentum_d2_adjust = 0.90;
double lrate_w = 0.0;
double lrate_b = 0.0;

bool is_gradient_checking = false;
bool use_log = false;
int training_epochs = 0;
int iter_per_epo = 0;

int tmpdebug = 0;

void
run(){
    std::vector<Mat> trainX;
    std::vector<Mat> testX;
    Mat trainY, testY;
    read_CIFAR10_data(trainX, testX, trainY, testY);

    std::vector<network_layer*> flow;
    buildNetworkFromConfigFile("config.txt", flow);

    trainNetwork(trainX, trainY, testX, testY, flow);

    trainX.clear();
    std::vector<Mat>().swap(trainX);
    testX.clear();
    std::vector<Mat>().swap(testX);
}

int 
main(int argc, char** argv){

    long start, end;
    start = clock();

    run();

/*
    Mat a = cv::Mat(1, 1, CV_64FC3, Scalar::all(1.0));
    cout<<a<<endl;

    Mat b = div(a, 2.0);
    //Mat b = a * 2.0;
//    Mat b = a.mul(Scalar::all(2.0));
    cout<<b<<endl;
//*/
/*
    Mat a = Mat::zeros(2, 2, CV_64FC3);
    a.AT3D(0, 0)[0] = 0.5;
    a.AT3D(0, 0)[1] = -0.5;
    a.AT3D(0, 0)[2] = 0.8;

    a.AT3D(0, 1)[0] = 1.5;
    a.AT3D(0, 1)[1] = 0.25;
    a.AT3D(0, 1)[2] = -0.5;

    a.AT3D(1, 0)[0] = 0.2;
    a.AT3D(1, 0)[1] = 0.6;
    a.AT3D(1, 0)[2] = -0.2;

    a.AT3D(1, 1)[0] = 1.0;
    a.AT3D(1, 1)[1] = 1.5;
    a.AT3D(1, 1)[2] = -2.5;

    cout<<" "<<a<<endl;

    Mat b = parallel3(nonLinearity, a, 2);
    cout<<" "<<b<<endl;

    Mat c = parallel3(dnonLinearity, b, 2);
    cout<<" "<<c<<endl;
*/

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}

