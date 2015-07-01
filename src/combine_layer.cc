#include "layer_bank.h"

using namespace std;

combine_layer::combine_layer(){}
combine_layer::~combine_layer(){
}

void combine_layer::init_config(string namestr, string outputformat){
    layer_type = "combine";
    layer_name = namestr;
    output_format = outputformat;
}

void combine_layer::forwardPass(int nsamples, network_layer* previous_layer){

}

void combine_layer::forwardPassTest(int nsamples, network_layer* previous_layer){

}

void combine_layer::init_weight(network_layer* previous_layer){

}


void combine_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){}

/*
void combine_layer::update(){}

*/




























