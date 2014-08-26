#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void getSample(vector<Mat>&, vector<Mat>&, int);
void getSample(Mat&, Mat&, int, int);
void getSample(vector<Mat>&, vector<Mat>&, Mat&, Mat&, int, int);