#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

typedef Mat (*func)(const Mat&);
typedef Mat (*func2)(const Mat&, const Mat&);
typedef Mat (*func3)(const Mat&, const Mat&, int);
typedef Mat (*func4)(const Mat&, int);
typedef Mat (*func5)(const Mat&, const Mat&, int, int, int);
typedef Mat (*func6)(const Mat&, int, int);

Mat parallel3(func, const Mat &);

Mat parallel3(func2, const Mat&, const Mat&);

Mat parallel3(func3, const Mat&, const Mat&, int);

Mat parallel3(func4, const Mat&, int);

Mat parallel3(func5, const Mat&, const Mat&, int, int, int);

Mat parallel3(func6, const Mat&, int, int);