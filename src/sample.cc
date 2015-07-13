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

    //cv::ocl::setUseOpenCL(true); 
    long start, end;
    start = clock();

    run();

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}

