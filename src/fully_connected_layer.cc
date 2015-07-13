#include "layer_bank.h"

using namespace std;

fully_connected_layer::fully_connected_layer(){}
fully_connected_layer::~fully_connected_layer(){
    
}

void fully_connected_layer::init_config(string namestr, int hiddensize, double weightDecay, string outputformat){
    layer_type = "fully_connected";
    layer_name = namestr;
    output_format = outputformat;
    size = hiddensize;
    weight_decay = weightDecay;
}

void fully_connected_layer::init_weight(network_layer* previous_layer){

    int inputsize = 0;
    if(previous_layer -> output_format == "image"){
        inputsize = previous_layer -> output_vector[0].size() * previous_layer -> output_vector[0][0].rows * previous_layer -> output_vector[0][0].cols * 3;
    }else{
        inputsize = previous_layer -> output_matrix.rows;
    }
    double epsilon = 0.12;
    w = Mat::ones(size, inputsize, CV_64FC1);
    randu(w, Scalar(-1.0), Scalar(1.0));
    w = w * epsilon;
    b = Mat::zeros(size, 1, CV_64FC1);
    wgrad = Mat::zeros(w.size(), CV_64FC1);
    bgrad = Mat::zeros(b.size(), CV_64FC1);
    wd2 = Mat::zeros(w.size(), CV_64FC1);
    bd2 = Mat::zeros(b.size(), CV_64FC1);
    
    // updater
    velocity_w = Mat::zeros(w.size(), CV_64FC1);
    velocity_b = Mat::zeros(b.size(), CV_64FC1);
    second_derivative_w = Mat::zeros(w.size(), CV_64FC1);
    second_derivative_b = Mat::zeros(b.size(), CV_64FC1);
    iter = 0;
    mu = 1e-2;
    fully_connected_layer::setMomentum();
}

void fully_connected_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void fully_connected_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) fully_connected_layer::setMomentum();
    second_derivative_w = momentum_second_derivative * second_derivative_w + (1.0 - momentum_second_derivative) * wd2;
    learning_rate = lrate_w / (second_derivative_w + mu);
    velocity_w = velocity_w * momentum_derivative + (1.0 - momentum_derivative) * wgrad.mul(learning_rate);
    w -= velocity_w;
    
    second_derivative_b = momentum_second_derivative * second_derivative_b + (1.0 - momentum_second_derivative) * bd2;
    learning_rate = lrate_b / (second_derivative_b + mu);
    velocity_b = velocity_b * momentum_derivative + (1.0 - momentum_derivative) * bgrad.mul(learning_rate);
    b -= velocity_b;
}

void fully_connected_layer::forwardPass(int nsamples, network_layer* previous_layer){
    Mat input;
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);    
    }else{
        previous_layer -> output_matrix.copyTo(input);
    }
    Mat tmpacti = w * input + repeat(b, 1, nsamples);
    tmpacti.copyTo(output_matrix);
}

void fully_connected_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    fully_connected_layer::forwardPass(nsamples, previous_layer);
}

void fully_connected_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    Mat input;
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix.copyTo(input);
    }

    if(next_layer -> output_format == "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
    }else{
        Mat derivative;
        Mat deriv2;
        next_layer -> delta_matrix.copyTo(derivative);
        next_layer -> d2_matrix.copyTo(deriv2);

        wgrad = derivative * input.t() / nsamples + weight_decay * w;
        bgrad = reduce(derivative, 1, CV_REDUCE_SUM) / nsamples;
        wd2 = deriv2 * pow(input.t(), 2.0) / nsamples + weight_decay;
        bd2 = reduce(deriv2, 1, CV_REDUCE_SUM) / nsamples;

        Mat tmp = w.t() * derivative;
        tmp.copyTo(delta_matrix);
        tmp = pow(w.t(), 2.0) * deriv2;
        tmp.copyTo(d2_matrix);
    }
}




























