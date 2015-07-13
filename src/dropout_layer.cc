#include "layer_bank.h"

using namespace std;

dropout_layer::dropout_layer(){}
dropout_layer::~dropout_layer(){
}

void dropout_layer::init_config(string namestr, string outputformat, double dor){
    layer_type = "dropout";
    layer_name = namestr;
    output_format = outputformat;
    dropout_rate = dor;
}

void dropout_layer::forwardPass(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        Mat input;
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix.copyTo(input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        Mat res = getBernoulliMatrix(input.rows, input.cols, dropout_rate);
        //cout<<"##############################"<<endl;
        //cout<<res<<endl;
        res.copyTo(bernoulli_matrix);
        res = res.mul(input);
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
        bernoulli_vector.clear();
        output_vector.resize(input.size());
        bernoulli_vector.resize(input.size());
        for(int i = 0; i < output_vector.size(); i++){
            output_vector[i].resize(input[i].size());
            bernoulli_vector[i].resize(input[i].size());
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
                vector<Mat> bnls(3);
                Mat bnl;
            	for(int ch = 0; ch < 3; ch++){
        			Mat tmp = getBernoulliMatrix(input[i][j].rows, input[i][j].cols, dropout_rate);
        			tmp.copyTo(bnls[ch]);
            	}
            	merge(bnls, bnl);
            	Mat res;
		        bnl.copyTo(bernoulli_vector[i][j]);
		        res = bnl.mul(input[i][j]);
		        res.copyTo(output_vector[i][j]);
            }
        }
        input.clear();
        std::vector<std::vector<Mat> >().swap(input);
    }
}

void dropout_layer::forwardPassTest(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        Mat input;
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix.copyTo(input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        Mat res;
        input.copyTo(res);
        res = res.mul(dropout_rate);
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
        bernoulli_vector.clear();
        output_vector.resize(input.size());
        bernoulli_vector.resize(input.size());
        for(int i = 0; i < output_vector.size(); i++){
            output_vector[i].resize(input[i].size());
            bernoulli_vector[i].resize(input[i].size());
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
                Mat res;
                input[i][j].copyTo(res);
                res = res.mul(Scalar::all(dropout_rate));
                res.copyTo(output_vector[i][j]);
            }
        }
        input.clear();
        std::vector<std::vector<Mat> >().swap(input);
    }
}

void dropout_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

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
        Mat tmp = derivative.mul(bernoulli_matrix);
        Mat tmp2 = deriv2.mul(pow(bernoulli_matrix, 2.0));
        tmp.copyTo(delta_matrix);
        tmp2.copyTo(d2_matrix);

    }else{
        if(previous_layer -> output_format != "image"){
            cout<<"??? image after matrix??? I can't do that for now..."<<endl;
            return;
        }
        std::vector<std::vector<Mat> > derivative;
        std::vector<std::vector<Mat> > deriv2;
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
                
            	Mat tmp = derivative[i][j].mul(bernoulli_vector[i][j]);
            	Mat tmp2 = deriv2[i][j].mul(pow(bernoulli_vector[i][j], 2.0));
		        tmp.copyTo(delta_vector[i][j]);
		        tmp2.copyTo(d2_vector[i][j]);
            }
        }
        derivative.clear();
        std::vector<std::vector<Mat> >().swap(derivative);
        deriv2.clear();
        std::vector<std::vector<Mat> >().swap(deriv2);
    }
}

/*
void dropout_layer::update(){}

void dropout_layer::init_weight(network_layer* previous_layer){}
*/




























