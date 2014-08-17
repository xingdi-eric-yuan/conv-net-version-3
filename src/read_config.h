#pragma once
#include "general_settings.h"
using namespace std;

string read_2_string(string);
bool get_word_bool(string&, string);
int get_word_int(string&, string);
double get_word_double(string&, string);
int get_word_type(string&, string);
void delete_comment(string&);
void delete_space(string &);
void get_layers_config(string&);
void readConfigFile(string);
