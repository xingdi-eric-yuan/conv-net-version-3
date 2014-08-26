#include "general_settings.h"
using namespace cv;
using namespace std;

std::vector<ConvLayerConfig> convConfig;
std::vector<FullConnectLayerConfig> fcConfig;
SoftmaxLayerConfig softmaxConfig;
///////////////////////////////////
// General parameters
///////////////////////////////////
bool is_gradient_checking = false;
bool use_log = false;
int log_iter = 0;
int batch_size = 1;
int pooling_method = 0;
int non_linearity = 2;
int training_epochs = 0;
double lrate_w = 0.0;
double lrate_b = 0.0;
int iter_per_epo = 0;
double lrate_decay = 0.0;

void
run(){
    vector<Mat> trainX;
    vector<Mat> testX;
    Mat trainY, testY;
    read_CIFAR10_data(trainX, testX, trainY, testY);
    int imgDim = trainX[0].rows;
    int nsamples = trainX.size();
    vector<Cvl> ConvLayers;
    vector<Fcl> HiddenLayers;
    Smr smr;
    readConfigFile("config.txt");
    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim, nsamples);
    // Train network using Back Propogation
    if(use_log){
        mkdir("log", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    trainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr);
    if(! is_gradient_checking){
        testNetwork(testX, testY, ConvLayers, HiddenLayers, smr);
    }
    ConvLayers.clear();
    HiddenLayers.clear();
}

int 
main(int argc, char** argv){
    string str = "clean_log";
    if(argv[1] && !str.compare(argv[1])){
        system("rm -rf log");
        cout<<"Cleaning log ..."<<endl;
        return 0;
    }
    long start, end;
    start = clock();

    run();

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}

