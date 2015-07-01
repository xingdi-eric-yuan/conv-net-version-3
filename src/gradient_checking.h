#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void gradient_checking(const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&, Mat&, Mat*);
  
void gradientChecking_SoftmaxLayer(std::vector<network_layer*>&, const std::vector<Mat>&, const Mat&);
void gradientChecking_FullyConnectedLayer(std::vector<network_layer*>&, const std::vector<Mat>&, const Mat&);
void gradientChecking_ConvolutionalLayer(std::vector<network_layer*>&, const std::vector<Mat>&, const Mat&);

void gradient_checking_network_layers(std::vector<network_layer*>&, const std::vector<Mat>&, const Mat&);