#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Scalar Reciprocal(const Scalar &);
Mat Reciprocal(const Mat &);
Mat sigmoid(const Mat &);
Mat dsigmoid_a(const Mat &);
Mat dsigmoid(const Mat &);
Mat ReLU(const Mat& );
Mat dReLU(const Mat& );
Mat Tanh(const Mat &);
Mat dTanh(const Mat &);
Mat nonLinearity(const Mat &);
Mat dnonLinearity(const Mat &);
Mat nonLinearityC3(const Mat &);
Mat dnonLinearityC3(const Mat &);
// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(const Mat &, int);
// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(const Mat&, const Mat&, int);
Mat convCalc(const Mat&, const Mat&, int);
// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat kron(const Mat&, const Mat&);
Mat getBernoulliMatrix(int, int, double);
double matNormalize(const Mat&, Mat*, double, double);
double matNormalizeUnsign(const Mat&, double, double);

// Follows are OpenCV maths
Mat exp(const Mat&);
Mat log(const Mat&);
Mat reduce(const Mat&, int, int);
Mat divide(const Mat&, const Mat&);
Mat pow(const Mat&, double);
double sum1(const Mat&);
double max(const Mat&);
double min(const Mat&);