#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void
//trainNetwork(vector<Mat> &, Mat &, vector<Cvl> &, vector<Fcl> &, Smr &);
trainNetwork(const vector<Mat> &, const Mat &, vector<Cvl> &, vector<Fcl> &, Smr &, const vector<Mat> &, const Mat &);
