#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

// int <==> string
string i2str(int);
int str2i(string);
//
Vec3d Scalar2Vec3d(Scalar);
Scalar Vec3d2Scalar(Vec3d);

void unconcatenateMat(const vector<Mat>&, vector<vector<Mat> >*, int);
Mat concatenateMat(const vector<vector<Mat> >&);
Mat concatenateMat(const vector<Mat>&, int );

double getLearningRate(const Mat&);
