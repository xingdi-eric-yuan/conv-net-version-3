#pragma once
#include "general_settings.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

void read_batch(string, vector<Mat>&, Mat&);
void read_CIFAR10_data(vector<Mat>&, vector<Mat>&, Mat&, Mat&);

Mat concat(const vector<Mat> &);
void preProcessing(vector<Mat>&, vector<Mat>&);

