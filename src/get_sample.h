#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void getSample(const vector<Mat>&, vector<Mat>*, int);
void getSample(const Mat&, Mat*, int, int);
void getSample(const vector<Mat>&, vector<Mat>*, const Mat&, Mat*, int, int);