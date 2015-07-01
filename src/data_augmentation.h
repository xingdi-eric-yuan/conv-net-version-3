#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;


void fliplr(const Mat &, Mat &);

void flipud(const Mat &, Mat &);

void flipudlr(const Mat &, Mat &);

void rotateNScale(const Mat &, Mat &, double, double);

void addWhiteNoise(const Mat &, Mat &, double);

void dataEnlarge(vector<Mat>&, Mat&);