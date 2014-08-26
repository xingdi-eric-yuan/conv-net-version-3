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

void unconcatenateMat(vector<Mat>&, vector<vector<Mat> >&, int);
Mat concatenateMat(vector<vector<Mat> >&);
Mat concatenateMat(vector<Mat>&, int );

double getLearningRate(Mat&);

void getSample(vector<Mat>&, vector<Mat>&, int);
void getSample(Mat&, Mat&, int);
void getSample(vector<Mat>&, vector<Mat>&, Mat&, Mat&, int);