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
#include "read_config.h"
#include "save_weights.h"
#include "get_sample.h"
#include "train_network.h"
#include "weight_init.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

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
// sample
#define SAMPLE_ROWS 0
#define SAMPLE_COLS 1

#define ATD at<double>
#define AT3D at<cv::Vec3d>
#define elif else if
#define $$LOG if(use_log && log_iter % 10 == 0){
#define $$_LOG }

using namespace std;
using namespace cv;

// Local Response Normalization
static int lrn_size = 5;
static double lrn_scale = 0.0000125;
static double lrn_beta = 0.75;

extern vector<ConvLayerConfig> convConfig;
extern vector<FullConnectLayerConfig> fcConfig;
extern SoftmaxLayerConfig softmaxConfig;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern bool is_gradient_checking;
extern bool use_log;
extern int log_iter;
extern int batch_size;
extern int pooling_method;
extern int non_linearity;
extern int training_epochs;
extern double lrate_w;
extern double lrate_b;
extern int iter_per_epo;
extern double lrate_decay;
