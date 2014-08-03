#pragma once
#include "general_settings.h"
#include <unordered_map>

using namespace std;
using namespace cv;
Point findLoc(Mat &, int );

vector<Point> findLocCh3(Mat &, int );
void minMaxLoc(Mat &, Scalar &, Scalar &, vector<Point> &, vector<Point> &);

Mat Pooling(Mat &, int , int , int , vector<vector<Point> > &, bool );

Mat UnPooling(Mat &, int , int , int , vector<vector<Point> > &);

Mat localResponseNorm(unordered_map<string, Mat> &, string , int);

void convAndPooling(vector<Mat> &, vector<Cvl> &, 
                unordered_map<string, Mat> &, 
                unordered_map<string, vector<vector<Point> > > &, bool);

void hashDelta(Mat &, unordered_map<string, Mat> &, vector<Cvl> &);