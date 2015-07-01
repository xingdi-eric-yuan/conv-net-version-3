#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "layer_bank.h"
#include "matrix_maths.h"
#include "channel_3.h"
#include "converter.h"
#include "read_config.h"
#include "read_data.h"
#include "train_network.h"
#include "gradient_checking.h"
#include "weights_IO.h"
#include "data_augmentation.h"

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
#define POOL_MEAN 1 
#define POOL_STOCHASTIC 2
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
#define NL_LEAKY_RELU 3

#define ATD at<double>
#define AT3D at<cv::Vec3d>
#define elif else if

using namespace std;
using namespace cv;

static double leaky_relu_alpha = 100.0;
extern bool is_gradient_checking;
extern bool use_log;
extern int training_epochs;
extern int iter_per_epo;
extern double lrate_w;
extern double lrate_b;

extern double momentum_w_init;
extern double momentum_d2_init;
extern double momentum_w_adjust;
extern double momentum_d2_adjust;

extern int tmpdebug;



