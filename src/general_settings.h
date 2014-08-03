#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "data_structure.h"
#include "convolution.h"
#include "string_proc.h"
#include "channel_3.h"
#include "cost_gradient.h"
#include "gradient_checking.h"
#include "helper.h"
#include "matrix_maths.h"
#include "read_data.h"
#include "result_predict.h"
#include "save_weights.h"
#include "train_network.h"
#include "weight_init.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

// Gradient Checking
#define G_CHECKING 1
// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1 //don't use
#define POOL_STOCHASTIC 2
// get Key type
#define KEY_CONV 0
#define KEY_POOL 1
#define KEY_DELTA 2
#define KEY_UP_DELTA 3
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

#define ATD at<double>
#define AT3D at<cv::Vec3d>
#define elif else if

using namespace std;
using namespace cv;
///////////////////////////////////
// General parameters
///////////////////////////////////
static bool DROPOUT = false;
static int nclasses = 10;
static int batch = 256;
static int Pooling_Methed = POOL_MAX;
static int nonlin = NL_RELU;

// Local Response Normalization
static int lrn_size = 5;
static double lrn_scale = 0.0000125;
static double lrn_beta = 0.75;

extern vector<ConvLayerConfig> convConfig;
extern vector<FullConnectLayerConfig> fcConfig;
