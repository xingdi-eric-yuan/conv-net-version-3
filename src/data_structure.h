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
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
}Cvl;

typedef struct FullConnectLayer{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
}Fcl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    double cost;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
struct ConvLayerConfig {
    int KernelSize;
    int KernelAmount;
    int PoolingDim;
    bool is3chKernel;
    bool useLRN; //LocalResponseNormalization
    ConvLayerConfig(int a, int b, int c, bool d, bool e) : KernelSize(a), KernelAmount(b), PoolingDim(c), is3chKernel(d) , useLRN(e){}
};

struct FullConnectLayerConfig {
    int NumHiddenNeurons;
    double DropoutRate;
    FullConnectLayerConfig(int a, double b) : NumHiddenNeurons(a), DropoutRate(b) {}
};

