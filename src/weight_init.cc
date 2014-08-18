#include "weight_init.h"

using namespace cv;
using namespace std;

void
weightRandomInit(ConvK &convk, int width, bool is3chKernel){
    if(is3chKernel){
        convk.W = Mat::ones(width, width, CV_64FC3);
        randu(convk.W, Scalar(-1.0, -1.0, -1.0), Scalar(1.0, 1.0, 1.0));
        convk.b = Scalar(0.0, 0.0, 0.0);
        convk.Wgrad = Mat::zeros(width, width, CV_64FC3);
        convk.bgrad = Scalar(0.0, 0.0, 0.0);
    }else{
        convk.W = Mat::ones(width, width, CV_64FC1);
        randu(convk.W, Scalar(-1.0), Scalar(1.0));
        convk.b = Scalar(0.0);
        convk.Wgrad = Mat::zeros(width, width, CV_64FC1);
        convk.bgrad = Scalar(0.0);
    }
    double epsilon = 0.12;
    convk.W = convk.W * epsilon;
}

void
weightRandomInit(Fcl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.012;
    ntw.W = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.W, Scalar(-1.0), Scalar(1.0));
    ntw.W = ntw.W * epsilon;
    ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.bgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W, Scalar(-1.0), Scalar(1.0));
    smr.W = smr.W * epsilon;
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
}

void
ConvNetInitPrarms(vector<Cvl> &ConvLayers, vector<Fcl> &HiddenLayers, Smr &smr, int imgDim, int nsamples){
    // Init Conv layers
    for(int i = 0; i < convConfig.size(); i++){
        Cvl tpcvl;
        for(int j = 0; j < convConfig[i].KernelAmount; j++){
            ConvK tmpConvK;
            weightRandomInit(tmpConvK, convConfig[i].KernelSize, convConfig[i].is3chKernel);
            tpcvl.layer.push_back(tmpConvK);
        }
        ConvLayers.push_back(tpcvl);
    }
    // Init Hidden layers
    int outDim = imgDim;
    for(int i = 0; i < convConfig.size(); i++){
        outDim = outDim - convConfig[i].KernelSize + 1;
        outDim = outDim / convConfig[i].PoolingDim;
    }
    int hiddenfeatures = pow(outDim, 2);
    for(int i = 0; i < ConvLayers.size(); i++){
        hiddenfeatures *= convConfig[i].KernelAmount;
    }
    Fcl tpntw;
    weightRandomInit(tpntw, hiddenfeatures * 3, fcConfig[0].NumHiddenNeurons);
    HiddenLayers.push_back(tpntw);
    for(int i = 1; i < fcConfig.size(); i++){
        Fcl tpntw2;
        weightRandomInit(tpntw2, fcConfig[i - 1].NumHiddenNeurons, fcConfig[i].NumHiddenNeurons);
        HiddenLayers.push_back(tpntw2);
    }
    // Init Softmax layer
    weightRandomInit(smr, softmaxConfig.NumClasses, fcConfig[fcConfig.size() - 1].NumHiddenNeurons);
}














