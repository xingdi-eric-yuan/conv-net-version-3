#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Scalar Reciprocal(Scalar &);
Mat Reciprocal(const Mat &);
Mat sigmoid(const Mat &);
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

// Follows are OpenCV maths
Mat exp(Mat);
Mat log(Mat);
Mat reduce(Mat, int, int);
Mat divide(Mat, Mat);
Mat pow(Mat, int);
double sum1(Mat);
double max(const Mat&);
double min(const Mat&);