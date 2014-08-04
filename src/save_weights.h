#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2txt3ch(const Mat &, string , int );
void save2txt(const Mat &, string , int);
void mkdir(const vector<Cvl> &);
void save2txt(const vector<Cvl> &, int);