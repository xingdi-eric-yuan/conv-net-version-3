#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2txt3ch(Mat &, string , int );
void save2txt(Mat &, string , int);
void mkdir(vector<Cvl> &);
void save2txt(vector<Cvl> &, int);