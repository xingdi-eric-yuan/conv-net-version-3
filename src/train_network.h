#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void forwardPassInit(const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&);
void forwardPass(const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&);
void forwardPassTest(const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&);

void backwardPass(std::vector<network_layer*>&);
void updateNetwork(std::vector<network_layer*>&, int);

void testNetwork(const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&);
void trainNetwork(const std::vector<Mat>&, const Mat&, const std::vector<Mat>&, const Mat&, std::vector<network_layer*>&);
