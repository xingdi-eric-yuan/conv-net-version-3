#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

int resultPredict(const Mat &, const vector<Cvl> &, const vector<Fcl> &, const Smr &);
Mat resultPredict(const vector<Mat> &, const vector<Cvl> &, const vector<Fcl> &, const Smr &);
void testNetwork(const vector<Mat> &, const Mat&, const vector<Cvl> &, const vector<Fcl> &, const Smr &);