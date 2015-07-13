#include "layer_bank.h"

using namespace std;

pooling_layer::pooling_layer(){}
pooling_layer::~pooling_layer(){
}
// with overlap
void pooling_layer::init_config(string namestr, int _method, string outputformat, int _stride, int windowsize){
    layer_type = "pooling";
    layer_name = namestr;
    output_format = outputformat;
    stride = _stride;
    window_size = windowsize;
    method = _method;
    overlap = true;
}
// without overlap
void pooling_layer::init_config(string namestr, int _method, string outputformat, int _stride){
    layer_type = "pooling";
    layer_name = namestr;
    output_format = outputformat;
    stride = _stride;
    method = _method;
    overlap = false;
    window_size = 0;
}

void pooling_layer::forwardPass(int nsamples, network_layer* previous_layer){
	
	if(previous_layer -> output_format == "matrix"){
        cout<<"??? Can not do pooling with matrix... give me an image..."<<endl;
        return;
	}
	std::vector<std::vector<Mat> > input(previous_layer -> output_vector);
    location.clear();
	output_vector.clear();
    location.resize(previous_layer -> output_vector.size());
    output_vector.resize(previous_layer -> output_vector.size());
    for(int i = 0; i < previous_layer -> output_vector.size(); i++){
        location[i].resize(previous_layer -> output_vector[i].size());
        output_vector[i].resize(previous_layer -> output_vector[i].size());
    }
	if(overlap){
		Size2i _size = Size(window_size, window_size);
		for(int i = 0; i < input.size(); i++){
			for(int j = 0; j < input[i].size(); j++){
				Mat tmp = Pooling_with_overlap(input[i][j], _size, stride, method, location[i][j]);
				tmp.copyTo(output_vector[i][j]);
			}
		}
	}else{
		for(int i = 0; i < input.size(); i++){
			for(int j = 0; j < input[i].size(); j++){
				Mat tmp = Pooling(input[i][j], stride, method, location[i][j]);
				tmp.copyTo(output_vector[i][j]);
			}
		}
	}
	input.clear();
	std::vector<std::vector<Mat> >().swap(input);
}

void pooling_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    
    if(previous_layer -> output_format == "matrix"){
        cout<<"??? Can not do pooling with matrix... give me an image..."<<endl;
        return;
    }
    std::vector<std::vector<Mat> > input(previous_layer -> output_vector);
    location.clear();
    output_vector.clear();
    location.resize(previous_layer -> output_vector.size());
    output_vector.resize(previous_layer -> output_vector.size());
    for(int i = 0; i < previous_layer -> output_vector.size(); i++){
        location[i].resize(previous_layer -> output_vector[i].size());
        output_vector[i].resize(previous_layer -> output_vector[i].size());
    }
    if(overlap){
        Size2i _size = Size(window_size, window_size);
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
                Mat tmp = Pooling_with_overlap_test(input[i][j], _size, stride, method);
                tmp.copyTo(output_vector[i][j]);
            }
        }
    }else{
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
                Mat tmp = Pooling_test(input[i][j], stride, method);
                tmp.copyTo(output_vector[i][j]);
            }
        }
    }
    input.clear();
    std::vector<std::vector<Mat> >().swap(input);
}

void pooling_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){
	

    if(previous_layer -> output_format != "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    std::vector<std::vector<Mat> > derivative;
    std::vector<std::vector<Mat> > deriv2;
    std::vector<std::vector<Mat> > input(previous_layer -> output_vector);
    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0].rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0].rows);
    }else{
        derivative = next_layer -> delta_vector;
        deriv2 = next_layer -> d2_vector;
    }
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(derivative.size());
    d2_vector.resize(derivative.size());
    for(int i = 0; i < delta_vector.size(); i++){
        delta_vector[i].resize(derivative[i].size());
        d2_vector[i].resize(derivative[i].size());
    }
	Size2i up_size = Size(input[0][0].rows, input[0][0].cols);
	Mat tmp;
	if(overlap){
		Size2i _size = Size(window_size, window_size);
	    for(int i = 0; i < derivative.size(); i++){
	        for(int j = 0; j < derivative[i].size(); j++){
				tmp = UnPooling_with_overlap(derivative[i][j], _size, stride, method, location[i][j], up_size);
				tmp.copyTo(delta_vector[i][j]);
				tmp = UnPooling_with_overlap(deriv2[i][j], _size, stride, method, location[i][j], up_size);
				tmp.copyTo(d2_vector[i][j]);
	        }
	    }
	}else{
	    for(int i = 0; i < derivative.size(); i++){
	        for(int j = 0; j < derivative[i].size(); j++){
				tmp = UnPooling(derivative[i][j], stride, method, location[i][j], up_size);
				tmp.copyTo(delta_vector[i][j]);
				tmp = UnPooling(deriv2[i][j], stride, method, location[i][j], up_size);
				tmp.copyTo(d2_vector[i][j]);
	        }
	    }
	}
    derivative.clear();
    std::vector<std::vector<Mat> >().swap(derivative);
    deriv2.clear();
    std::vector<std::vector<Mat> >().swap(deriv2);
    input.clear();
    std::vector<std::vector<Mat> >().swap(input);
}

/*

void pooling_layer::update(){}

void pooling_layer::init_weight(network_layer* previous_layer){}

*/




























