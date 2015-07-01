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
Scalar makeScalar(double);

void convert(std::vector<std::vector<Mat> >&, Mat&);
void convert(Mat&, std::vector<std::vector<Mat> >&, int, int);


/*
void unconcatenateMat(const vector<Mat>&, vector<vector<Mat> >*, int);
Mat concatenateMat(const vector<vector<Mat> >&);
Mat concatenateMat(const vector<Mat>&, int );

void splitChannels(vector<vector<Mat> >&);

double getLearningRate(const Mat&);


*/