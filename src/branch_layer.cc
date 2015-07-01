#include "layer_bank.h"

using namespace std;

branch_layer::branch_layer(){}
branch_layer::~branch_layer(){
}

void branch_layer::init_config(string namestr, string outputformat){
    layer_type = "branch";
    layer_name = namestr;
    output_format = outputformat;
}

void branch_layer::forwardPass(int nsamples, network_layer* previous_layer){

}

void branch_layer::forwardPassTest(int nsamples, network_layer* previous_layer){

}

void branch_layer::init_weight(network_layer* previous_layer){

}


void branch_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){}

/*
void branch_layer::update(){}

*/




























