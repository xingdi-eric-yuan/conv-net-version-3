#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void gradientChecking_ConvLayer(vector<Cvl>&, vector<Fcl>&, Smr&, vector<Mat>&, Mat&, double);

void gradientChecking_FullConnectLayer(vector<Cvl>&, vector<Fcl>&, Smr&, vector<Mat>&, Mat&, double);

void gradientChecking_SoftmaxLayer(vector<Cvl>&, vector<Fcl>&, Smr&, vector<Mat>&, Mat&, double);
