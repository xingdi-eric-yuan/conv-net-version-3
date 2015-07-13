#include "layer_bank.h"

using namespace std;

softmax_layer::softmax_layer(){}
softmax_layer::~softmax_layer(){
}

void softmax_layer::init_config(string namestr, int numclasses, double weightDecay, string outputformat){
    layer_type = "softmax";
    layer_name = namestr;
    output_format = outputformat;
    output_size = numclasses;
    weight_decay = weightDecay;
}

void softmax_layer::init_weight(network_layer* previous_layer){

    int inputsize = 0;
    if(previous_layer -> output_format == "image"){
        inputsize = previous_layer -> output_vector[0].size() * previous_layer -> output_vector[0][0].rows * previous_layer -> output_vector[0][0].cols * 3;
    }else{
        inputsize = previous_layer -> output_matrix.rows;
    }
    double epsilon = 0.12;
    w = Mat::ones(output_size, inputsize, CV_64FC1);
    randu(w, Scalar(-1.0), Scalar(1.0));
    w = w * epsilon;
    b = Mat::zeros(output_size, 1, CV_64FC1);
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
    softmax_layer::setMomentum();
}

void softmax_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void softmax_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) softmax_layer::setMomentum();
    second_derivative_w = momentum_second_derivative * second_derivative_w + (1.0 - momentum_second_derivative) * wd2;
    learning_rate = lrate_w / (second_derivative_w + mu);
    velocity_w = velocity_w * momentum_derivative + (1.0 - momentum_derivative) * wgrad.mul(learning_rate);
    w -= velocity_w;
    
    second_derivative_b = momentum_second_derivative * second_derivative_b + (1.0 - momentum_second_derivative) * bd2;
    learning_rate = lrate_b / (second_derivative_b + mu);
    velocity_b = velocity_b * momentum_derivative + (1.0 - momentum_derivative) * bgrad.mul(learning_rate);
    b -= velocity_b;
}

void softmax_layer::forwardPass(int nsamples, network_layer* previous_layer){
    Mat input;
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix.copyTo(input);
    }
    Mat M = w * input + repeat(b, 1, nsamples);
    M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
    M = exp(M);
    Mat p = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
    p.copyTo(output_matrix);
}

void softmax_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    Mat input;
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix.copyTo(input);
    }
    Mat M = w * input + repeat(b, 1, nsamples);
    M.copyTo(output_matrix);
}

void softmax_layer::backwardPass(int nsamples, network_layer* previous_layer, Mat& groundTruth){

    Mat input;
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix.copyTo(input);
    }
    Mat derivative = groundTruth - output_matrix;
    wgrad = -derivative * input.t() / nsamples + weight_decay * w;
    bgrad = -reduce(derivative, 1, CV_REDUCE_SUM) / nsamples;
    wd2 = pow(derivative, 2.0) * pow(input.t(), 2.0) / nsamples + weight_decay;
    bd2 = reduce(pow(derivative, 2.0), 1, CV_REDUCE_SUM) / nsamples;

    Mat tmp = -w.t() * derivative;
    tmp.copyTo(delta_matrix);
    tmp = pow(w.t(), 2.0) * pow(derivative, 2.0);
    tmp.copyTo(d2_matrix);

}

























