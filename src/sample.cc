#include "general_settings.h"
using namespace cv;
using namespace std;

std::vector<ConvLayerConfig> convConfig;
std::vector<FullConnectLayerConfig> fcConfig;

void
run(){
    vector<Mat> trainX;
    vector<Mat> testX;
    Mat trainY, testY;

    read_CIFAR10_data(trainX, testX, trainY, testY);

    cout<<"Read trainX successfully, including "<<trainX[0].cols * trainX[0].rows<<" features and "<<trainX.size()<<" samples."<<endl;
    cout<<"Read trainY successfully, including "<<trainY.cols<<" samples"<<endl;
    cout<<"Read testX successfully, including "<<testX[0].cols * testX[0].rows<<" features and "<<testX.size()<<" samples."<<endl;
    cout<<"Read testY successfully, including "<<testY.cols<<" samples"<<endl;
    int amount;
    if(G_CHECKING){
        amount = 20;
    }else{
        amount = 50000;
    }
    vector<Mat> tp;
    for(int i = 0; i < amount; i++){
        tp.push_back(trainX[i]);
    }
    Rect roi = Rect(0, 0, amount, trainY.rows);
    trainY = trainY(roi);

    int imgDim = tp[0].rows;
    int nsamples = tp.size();
    vector<Cvl> ConvLayers;
    vector<Fcl> HiddenLayers;
    Smr smr;

    convConfig.push_back(ConvLayerConfig(18, 2, 5, false, true));
    fcConfig.push_back(FullConnectLayerConfig(100, 0.5));

    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim, nsamples);
    // Train network using Back Propogation
    trainNetwork(tp, trainY, ConvLayers, HiddenLayers, smr, 1e-3);

    if(! G_CHECKING){
        // Test use test set
        Mat result = resultProdict(testX, ConvLayers, HiddenLayers, smr, 1e-3);
        Mat err(testY);
        err -= result;
        int correct = err.cols;
        for(int i=0; i<err.cols; i++){
            if(err.ATD(0, i) != 0) --correct;
        }
        cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
        result.release();
        err.release();
    }    
    ConvLayers.clear();
    HiddenLayers.clear();
}

int 
main(int argc, char** argv){
    
    long start, end;
    start = clock();

    run();

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}











