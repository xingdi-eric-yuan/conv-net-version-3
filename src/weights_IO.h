#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2XML(string, string, std::vector<network_layer*>&);

// void readFromXML(string, std::vector<network_layer*>&);
