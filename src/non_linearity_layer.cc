#include "layer_bank.h"

using namespace std;

non_linearity_layer::non_linearity_layer(){}
non_linearity_layer::~non_linearity_layer(){
}

void non_linearity_layer::init_config(string namestr, int _method, string outputformat){
    layer_type = "non_linearity";
    layer_name = namestr;
    method = _method;
    output_format = outputformat;
}

void non_linearity_layer::forwardPass(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        Mat input;
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix.copyTo(input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        Mat res = nonLinearity(input, method);
        res.copyTo(output_matrix);
    }else{ // output_format == "image"

        std::vector<std::vector<Mat> > input;
        if(previous_layer -> output_format == "image"){
            input = previous_layer -> output_vector;
        }else{
            cout<<"??? image after matrix??? I can't do that for now..."<<endl;
            return;
        }

        output_vector.clear();
        output_vector.resize(input.size());
        for(int i = 0; i < output_vector.size(); i++){
            output_vector[i].resize(input[i].size());
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
                Mat res = parallel3(nonLinearity, input[i][j], method);
                res.copyTo(output_vector[i][j]);
            }
        }
        input.clear();
        std::vector<std::vector<Mat> >().swap(input);
    }
}

void non_linearity_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    non_linearity_layer::forwardPass(nsamples, previous_layer);
}

void non_linearity_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    if(output_format == "matrix"){

        Mat derivative;
        Mat deriv2;
        if(next_layer -> output_format == "matrix"){
            next_layer -> delta_matrix.copyTo(derivative);
            next_layer -> d2_matrix.copyTo(deriv2);
        }else{
            convert(next_layer -> delta_vector, derivative);
            convert(next_layer -> d2_vector, deriv2);
        }
        Mat input;
        if(previous_layer -> output_format == "image"){
            convert(previous_layer -> output_vector, input);
        }else{
            previous_layer -> output_matrix.copyTo(input);
        }
        Mat tmp = dnonLinearity(input, method);
        Mat tmp2 = derivative.mul(tmp);
        tmp2.copyTo(delta_matrix);

        tmp2 = deriv2.mul(pow(tmp, 2.0));
        tmp2.copyTo(d2_matrix);
    }else{
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
        for(int i = 0; i < derivative.size(); i++){
            for(int j = 0; j < derivative[i].size(); j++){
                Mat res = parallel3(dnonLinearity, input[i][j], method);
                Mat tmp = derivative[i][j].mul(res);
                tmp.copyTo(delta_vector[i][j]);
                tmp = deriv2[i][j].mul(pow(res, 2.0));
                tmp.copyTo(d2_vector[i][j]);
            }
        }
        derivative.clear();
        std::vector<std::vector<Mat> >().swap(derivative);
        deriv2.clear();
        std::vector<std::vector<Mat> >().swap(deriv2);
        input.clear();
        std::vector<std::vector<Mat> >().swap(input);

    }
}
/*
void non_linearity_layer::update(){}

*/




























