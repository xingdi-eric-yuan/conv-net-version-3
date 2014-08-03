#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Scalar Reciprocal(Scalar &);
Mat Reciprocal(Mat &);
Mat sigmoid(Mat &);
Mat dsigmoid(Mat &);
Mat ReLU(Mat& );
Mat dReLU(Mat& );
Mat Tanh(Mat &);
Mat dTanh(Mat &);
Mat nonLinearity(Mat &);
Mat dnonLinearity(Mat &);
Mat nonLinearityC3(Mat &);
Mat dnonLinearityC3(Mat &);
// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(Mat &, int);
// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(Mat &, Mat &, int);
Mat convCalc(Mat &, Mat &, int );
// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat kron(Mat&, Mat&);
Mat getBernoulliMatrix(int, int, double);
