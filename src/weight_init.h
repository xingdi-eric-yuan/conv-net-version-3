#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void weightRandomInit(ConvK&, int, bool);

void weightRandomInit(Fcl&, int, int);

void weightRandomInit(Smr&, int, int);

void ConvNetInitPrarms(vector<Cvl>&, vector<Fcl>&, Smr&, int, int);