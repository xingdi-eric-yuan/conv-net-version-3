#pragma once
#include "general_settings.h"
#include <unordered_map>

using namespace std;
using namespace cv;
Point findLoc(Mat &, int );

vector<Point> findLocCh3(Mat &, int );
void minMaxLoc(Mat &, Scalar &, Scalar &, vector<Point> &, vector<Point> &);

Mat Pooling(const Mat &, int , int , int , vector<vector<Point> > &, bool );

Mat UnPooling(const Mat &, int , int , int , vector<vector<Point> > &);

Mat localResponseNorm(const unordered_map<string, Mat> &, string);

Mat dlocalResponseNorm(const unordered_map<string, Mat> &, string);

void convAndPooling(const vector<Mat> &, const vector<Cvl> &, 
                unordered_map<string, Mat> &, 
                unordered_map<string, vector<vector<Point> > > &, bool);

void hashDelta(const Mat &, unordered_map<string, Mat> &, vector<Cvl> &, int);