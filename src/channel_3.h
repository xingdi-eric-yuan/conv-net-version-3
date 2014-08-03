#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

typedef Mat (*func)(Mat&);
typedef Mat (*func2)(Mat&, Mat&);
typedef Mat (*func3)(Mat&, Mat&, int);
typedef Mat (*func4)(Mat&, int);

Mat parallel3(func, Mat &);

Mat parallel3(func2, Mat&, Mat&);

Mat parallel3(func3, Mat&, Mat&, int);

Mat parallel3(func4, Mat&, int);