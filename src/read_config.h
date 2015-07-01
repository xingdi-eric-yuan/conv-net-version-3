#pragma once
#include "general_settings.h"
using namespace std;

class network_layer;

string read_2_string(string);
bool get_word_bool(string&, string);
int get_word_int(string&, string);
string get_word_string(string&, string);
double get_word_double(string&, string);
int get_word_type(string&, string);
void delete_comment(string&);
void delete_space(string &);
void get_layers_config(string&, std::vector<network_layer*> &);
void buildNetworkFromConfigFile(string, std::vector<network_layer*> &);
