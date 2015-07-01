#include "layer_bank.h"

using namespace std;

// kernel
network_layer::network_layer(){}
network_layer::~network_layer(){
    output_matrix.release();
    output_vector.clear();
    std::vector<std::vector<Mat> >().swap(output_vector);
}










