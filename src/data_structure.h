#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;
///////////////////////////////////
// Network Layer Structures
///////////////////////////////////
typedef struct ConvKernel{
    Mat W;
    Scalar b;
    Mat Wgrad;
    Scalar bgrad;
    Mat Wd2;
    Scalar bd2;
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
}Cvl;

typedef struct FullConnectLayer{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    Mat Wd2;
    Mat bd2;
}Fcl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    double cost;
    Mat Wd2;
    Mat bd2;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
struct ConvLayerConfig {
    int KernelSize;
    int KernelAmount;
    double WeightDecay;
    int PoolingDim;
    bool is3chKernel;
    bool useLRN; //LocalResponseNormalization
    ConvLayerConfig(int a, int b, double c, int d, bool e, bool f) : KernelSize(a), KernelAmount(b), WeightDecay(c), PoolingDim(d), is3chKernel(e) , useLRN(f){}
};

struct FullConnectLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    double DropoutRate;
    FullConnectLayerConfig(int a, double b, double c) : NumHiddenNeurons(a), WeightDecay(b), DropoutRate(c) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
};
