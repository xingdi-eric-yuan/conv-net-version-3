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
Mat nonLinearity(const Mat &, int);
Mat dnonLinearity(const Mat &, int);
// Mimic rot90() in Matlab/GNU Octave.
Mat rot90(const Mat &, int);
// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat conv2(const Mat&, const Mat&, int, int, int);

Mat convolveDFT(const Mat&, const Mat&);
Mat conv2dft(const Mat&, const Mat&, int, int, int);

Mat convCalc(const Mat&, const Mat&, int, int, int);
Mat doPadding(Mat&, int);
Mat dePadding(Mat&, int);
Mat interpolation(Mat&, Mat&);
Mat interpolation(Mat&, int);
// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat kron(const Mat&, const Mat&);
Mat getBernoulliMatrix(int, int, double);
//double matNormalize(const Mat&, Mat*, double, double);
//double matNormalizeUnsign(const Mat&, double, double);

// Follows are OpenCV maths
Mat div(double, const Mat&);
Mat div(const Mat&, double);
Scalar div(const Scalar&, double);
Mat exp(const Mat&);
Mat log(const Mat&);
Mat reduce(const Mat&, int, int);
Mat divide(const Mat&, const Mat&);
Mat pow(const Mat&, double);
double sum1(const Mat&);
double max(const Mat&);
double min(const Mat&);

Mat Pooling_with_overlap(const Mat&, Size2i, int, int, std::vector<std::vector<Point> >&);
Mat Pooling_with_overlap_test(const Mat&, Size2i, int, int);
Mat Pooling(const Mat&, int, int, std::vector<std::vector<Point> > &);
Mat Pooling_test(const Mat&, int, int);
Mat UnPooling(const Mat&, int, int, std::vector<std::vector<Point> >&, Size2i);
Mat UnPooling_with_overlap(const Mat&, Size2i, int, int, std::vector<std::vector<Point> >&, Size2i);

Point findLoc(const Mat&, int);
std::vector<Point> findLocCh3(const Mat&, int);
Mat findMax(const Mat&);
void minMaxLoc(const Mat&, Scalar&, Scalar&, std::vector<Point>&, std::vector<Point>&);

int compareMatrix(const Mat&, const Mat&);





