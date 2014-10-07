#pragma once
#include "general_settings.h"
#include <unordered_map>

using namespace std;
using namespace cv;

#include "convolution.h"


Point findLoc(const Mat &, int);
vector<Point> findLocCh3(const Mat &, int);

void minMaxLoc(const Mat &, Scalar &, Scalar &, vector<Point> &, vector<Point> &);
Mat Pooling(const Mat &, int , int , int , vector<vector<Point> > &);

Mat Pooling(const Mat &, int , int , int );


Mat UnPooling(const Mat &, int , int , int , vector<vector<Point> > &);
Mat localResponseNorm(const unordered_map<string, Mat> &, string );
Mat localResponseNorm(const vector<vector<Mat> > &, int , int , int , int );
Mat dlocalResponseNorm(const unordered_map<string, Mat> &, string );

void convAndPooling(const vector<Mat> &, const vector<Cvl> &, 
                unordered_map<string, Mat> &, 
                unordered_map<string, vector<vector<Point> > >&);

void hashDelta(const Mat &, unordered_map<string, Mat> &, int , int );
void convAndPooling(const vector<Mat> &, const vector<Cvl> &, vector<vector<Mat> > &);