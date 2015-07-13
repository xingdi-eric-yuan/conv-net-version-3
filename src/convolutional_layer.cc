#include "layer_bank.h"

using namespace std;

// kernel
convolutional_kernel::convolutional_kernel(){}
convolutional_kernel::~convolutional_kernel(){
//    w.release();
//    wgrad.release();
//    wd2.release();
}

void convolutional_kernel::init_config(int width, double weightDecay){
    kernel_size = width;
    weight_decay = weightDecay;
    w = cv::Mat(kernel_size, kernel_size, CV_64FC3, Scalar::all(1.0));
    randu(w, Scalar::all(-1.0), Scalar::all(1.0));
    b = Scalar::all(0.0);
    wgrad = Mat::zeros(w.size(), CV_64FC3);
    bgrad = Scalar::all(0.0);
    wd2 = Mat::zeros(w.size(), CV_64FC3);
    bd2 = Scalar::all(0.0);

    double epsilon = 0.12;
    w = w * epsilon;
}

// layer 
convolutional_layer::convolutional_layer(){}
convolutional_layer::~convolutional_layer(){
    kernels.clear();
    vector<convolutional_kernel*>().swap(kernels);
}

void convolutional_layer::init_config(string namestr, int kernel_amount, int kernel_size, int output_amount, int _padding, int _stride, double weight_decay, string outputformat){
    layer_type = "convolutional";
    layer_name = namestr;
    output_format = outputformat;
    padding = _padding;
    stride = _stride;
    combine_feature_map = output_amount;

    kernels.clear();
    for(int i = 0; i < kernel_amount; ++i){
        convolutional_kernel *tmp_kernel = new convolutional_kernel();
        tmp_kernel -> init_config(kernel_size, weight_decay);
        kernels.push_back(tmp_kernel);
    }
}

void convolutional_layer::init_weight(network_layer* previous_layer){

    if(combine_feature_map > 0){
        combine_weight = Mat::ones(kernels.size(), combine_feature_map, CV_64FC1);
    //    randu(combine_weight, Scalar(-1.0), Scalar(1.0));
    //    combine_weight = combine_weight.mul(0.12);
        combine_weight_grad = Mat::zeros(combine_weight.size(), CV_64FC1);
        combine_weight_d2 = Mat::zeros(combine_weight.size(), CV_64FC1);
    }

    // updater
    Mat tmpw = Mat::zeros(kernels[0] -> w.size(), CV_64FC3);
    velocity_combine_weight = Mat::zeros(combine_weight.size(), CV_64FC1);
    second_derivative_combine_weight = Mat::zeros(combine_weight.size(), CV_64FC1);

    velocity_w.resize(kernels.size());
    velocity_b.resize(kernels.size());
    second_derivative_w.resize(kernels.size());
    second_derivative_b.resize(kernels.size());
    for(int i = 0; i < kernels.size(); ++i){
        tmpw.copyTo(velocity_w[i]);
        tmpw.copyTo(second_derivative_w[i]);
        velocity_b[i] = Scalar::all(0.0);
        second_derivative_b[i] = Scalar::all(0.0);
    }
    iter = 0;
    mu = 1e-2;
    convolutional_layer::setMomentum();
    tmpw.release();
}

void convolutional_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void convolutional_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) convolutional_layer::setMomentum();
    
    for(int i = 0; i < kernels.size(); ++i){
        second_derivative_w[i] = momentum_second_derivative * second_derivative_w[i] + (1.0 - momentum_second_derivative) * kernels[i] -> wd2;
        learning_rate_w = lrate_w / (second_derivative_w[i] + Scalar::all(mu));
        velocity_w[i] = velocity_w[i] * momentum_derivative + (1.0 - momentum_derivative) * kernels[i] -> wgrad.mul(learning_rate_w);
        kernels[i] -> w -= velocity_w[i];

        second_derivative_b[i] = momentum_second_derivative * second_derivative_b[i] + (1.0 - momentum_second_derivative) * kernels[i] -> bd2;
        learning_rate_b = lrate_b / (second_derivative_b[i] + Scalar::all(mu));
        velocity_b[i] = velocity_b[i] * momentum_derivative + (1.0 - momentum_derivative) * kernels[i] -> bgrad.mul(learning_rate_b);
        kernels[i] -> b -= velocity_b[i];
    }
    if(combine_feature_map > 0){
        second_derivative_combine_weight = momentum_second_derivative * second_derivative_combine_weight + (1.0 - momentum_second_derivative) * combine_weight_d2;
        learning_rate_w = lrate_w / (second_derivative_combine_weight + mu);
        velocity_combine_weight = velocity_combine_weight * momentum_derivative + (1.0 - momentum_derivative) * combine_weight_grad.mul(learning_rate_w);
        combine_weight -= velocity_combine_weight;
    }
}

void convolutional_layer::forwardPass(int nsamples, network_layer* previous_layer){

    std::vector<std::vector<Mat> > input;
    if(previous_layer -> output_format == "image"){
        input = previous_layer -> output_vector;
    }else{
        // no!!!!
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    Mat c_weight;
    if(combine_feature_map > 0){
        c_weight = exp(combine_weight);
        c_weight = divide(c_weight, repeat(reduce(c_weight, 0, CV_REDUCE_SUM), c_weight.rows, 1));
    }
    output_vector.clear();
    for(int i = 0; i < input.size(); ++i){
        std::vector<Mat> eachsample;
        for(int j = 0; j < input[i].size(); ++j){
            std::vector<Mat> tmpvec;
            for(int k = 0; k < kernels.size(); ++k){
                Mat temp = rot90(kernels[k] -> w, 2);
                Mat tmpconv = convCalc(input[i][j], temp, CONV_VALID, padding, stride);
                tmpconv += kernels[k] -> b;
                tmpvec.push_back(tmpconv);
            }
            if(combine_feature_map > 0){
                std::vector<Mat> outputvec(combine_feature_map);
                Mat zero = Mat::zeros(tmpvec[0].size(), CV_64FC3);
                for(int k = 0; k < outputvec.size(); k++) {zero.copyTo(outputvec[k]);}
                for(int m = 0; m < kernels.size(); m++){
                    for(int n = 0; n < combine_feature_map; n++){
                        outputvec[n] += tmpvec[m].mul(Scalar::all(c_weight.ATD(m, n)));
                    }
                }
                for(int k = 0; k < outputvec.size(); k++) {eachsample.push_back(outputvec[k]);}
                outputvec.clear();
            }
            else{
                for(int k = 0; k < tmpvec.size(); k++) {eachsample.push_back(tmpvec[k]);}
            }
            tmpvec.clear();
        }
        output_vector.push_back(eachsample);
    }
    input.clear();
    std::vector<std::vector<Mat> >().swap(input);
    c_weight.release();
}

void convolutional_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    convolutional_layer::forwardPass(nsamples, previous_layer);
}

void convolutional_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    std::vector<std::vector<Mat> > derivative;
    std::vector<std::vector<Mat> > deriv2;
    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0].rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0].rows);
    }else{
        derivative = next_layer -> delta_vector;
        deriv2 = next_layer -> d2_vector;
    }
    if(previous_layer -> output_format != "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(previous_layer -> output_vector.size());
    d2_vector.resize(previous_layer -> output_vector.size());
    for(int i = 0; i < delta_vector.size(); i++){
        delta_vector[i].resize(previous_layer -> output_vector[i].size());
        d2_vector[i].resize(previous_layer -> output_vector[i].size());
    }

    Mat tmp, tmp2, tmp3;
    std::vector<Mat> tmp_wgrad(kernels.size());
    std::vector<Mat> tmp_wd2(kernels.size());
    std::vector<Scalar> tmpgradb;
    std::vector<Scalar> tmpbd2;
    tmp = Mat::zeros(kernels[0] -> w.size(), CV_64FC3);
    Scalar tmpscalar(0.0, 0.0, 0.0);
    for(int m = 0; m < kernels.size(); m++) {
        tmp.copyTo(tmp_wgrad[m]);
        tmp.copyTo(tmp_wd2[m]);
        tmpgradb.push_back(tmpscalar);
        tmpbd2.push_back(tmpscalar);
    }

    Mat c_weight, c_weightgrad, c_weightd2;
    if(combine_feature_map > 0){
        c_weight = exp(combine_weight);
        c_weight = divide(c_weight, repeat(reduce(c_weight, 0, CV_REDUCE_SUM), c_weight.rows, 1));
        c_weightgrad = Mat::zeros(c_weight.size(), CV_64FC1);
        c_weightd2 = Mat::zeros(c_weight.size(), CV_64FC1);
    }

    for(int i = 0; i < nsamples; i++){
        for(int j = 0; j < previous_layer -> output_vector[i].size(); j++){
            std::vector<Mat> sensi(kernels.size());
            std::vector<Mat> sensid2(kernels.size());
            Mat tmp_delta;
            Mat tmp_d2;
            tmp = Mat::zeros(output_vector[0][0].size(), CV_64FC3);
            for(int m = 0; m < kernels.size(); m++) {

                tmp.copyTo(sensi[m]);
                tmp.copyTo(sensid2[m]);
                if(combine_feature_map > 0){
                    for(int n = 0; n < combine_feature_map; n++){
                        sensi[m] += derivative[i][j * combine_feature_map + n].mul(Scalar::all(c_weight.ATD(m, n)));
                        sensid2[m] += deriv2[i][j * combine_feature_map + n].mul(Scalar::all(pow(c_weight.ATD(m, n), 2.0)));
                    }
                }else{
                    sensi[m] += derivative[i][j * kernels.size() + m];
                    sensid2[m] += deriv2[i][j * kernels.size() + m];
                }

                if(stride > 1){
                    int len = previous_layer -> output_vector[0][0].rows + padding * 2 - kernels[0] -> w.rows + 1;
                    sensi[m] = interpolation(sensi[m], len);
                    sensid2[m] = interpolation(sensid2[m], len);
                }

                if(m == 0){
                    tmp_delta = convCalc(sensi[m], kernels[m] -> w, CONV_FULL, 0, 1);
                    tmp_d2 = convCalc(sensid2[m], pow(kernels[m] -> w, 2.0), CONV_FULL, 0, 1);
                }else{
                    tmp_delta += convCalc(sensi[m], kernels[m] -> w, CONV_FULL, 0, 1);
                    tmp_d2 += convCalc(sensid2[m], pow(kernels[m] -> w, 2.0), CONV_FULL, 0, 1);
                }
                Mat input;
                if(padding > 0){
                    input = doPadding(previous_layer -> output_vector[i][j], padding);
                }else{
                    previous_layer -> output_vector[i][j].copyTo(input);
                }
                tmp2 = rot90(sensi[m], 2);
                tmp3 = rot90(sensid2[m], 2);
                tmp_wgrad[m] += convCalc(input, tmp2, CONV_VALID, 0, 1);
                tmp_wd2[m] += convCalc(pow(input, 2.0), tmp3, CONV_VALID, 0, 1);
                tmpgradb[m] += sum(tmp2);
                tmpbd2[m] += sum(tmp3);

                if(combine_feature_map > 0){
                    // combine feature map weight matrix (after softmax)
                    previous_layer -> output_vector[i][j].copyTo(input);
                    tmp2 = rot90(kernels[m] -> w, 2);
                    tmp2.copyTo(tmp3);
                    tmp2 = convCalc(input, tmp2, CONV_VALID, padding, stride);
                    tmp3 = convCalc(pow(input, 2.0), pow(tmp3, 2.0), CONV_VALID, padding, stride);
                    for(int n = 0; n < combine_feature_map; n++){
                        Mat tmpd;
                        tmpd = tmp2.mul(derivative[i][j * combine_feature_map + n]);
                        c_weightgrad.ATD(m, n) += sum1(tmpd);
                        tmpd = tmp3.mul(deriv2[i][j * combine_feature_map + n]);
                        c_weightd2.ATD(m, n) += sum1(tmpd);
                    }
                }
            }
            if(padding > 0){
                tmp_delta = dePadding(tmp_delta, padding);
                tmp_d2 = dePadding(tmp_d2, padding);
            }
            tmp_delta.copyTo(delta_vector[i][j]);
            tmp_d2.copyTo(d2_vector[i][j]);
            sensi.clear();
            std::vector<Mat>().swap(sensi);
            sensid2.clear();
            std::vector<Mat>().swap(sensid2);
        }
    }

    for(int i = 0; i < kernels.size(); i++){
        kernels[i] -> wgrad = div(tmp_wgrad[i], nsamples) + kernels[i] -> w * kernels[i] -> weight_decay;
        kernels[i] -> wd2 = div(tmp_wd2[i], nsamples) + Scalar::all(kernels[i] -> weight_decay);
        kernels[i] -> bgrad = div(tmpgradb[i], nsamples);
        kernels[i] -> bd2 = div(tmpbd2[i], nsamples);
    }

    if(combine_feature_map > 0){
        tmp2 = c_weightgrad.mul(c_weight);
        tmp2 = repeat(reduce(tmp2, 0, CV_REDUCE_SUM), c_weightgrad.rows, 1);
        tmp = c_weightgrad - tmp2;
        tmp = c_weight.mul(tmp);
        tmp = div(tmp, nsamples);
        tmp.copyTo(combine_weight_grad);

        tmp2 = c_weightd2.mul(c_weight);
        tmp2 = repeat(reduce(tmp2, 0, CV_REDUCE_SUM), c_weightd2.rows, 1);
        tmp = c_weightd2 - tmp2;
        tmp = c_weight.mul(tmp);
        tmp = div(tmp, nsamples);
        tmp.copyTo(combine_weight_d2);
    }
    tmp_wgrad.clear();
    std::vector<Mat>().swap(tmp_wgrad);
    tmp_wd2.clear();
    std::vector<Mat>().swap(tmp_wd2);
    derivative.clear();
    std::vector<std::vector<Mat> >().swap(derivative);
    deriv2.clear();
    std::vector<std::vector<Mat> >().swap(deriv2);
    tmpgradb.clear();
    std::vector<Scalar>().swap(tmpgradb);
    tmpbd2.clear();
    std::vector<Scalar>().swap(tmpbd2);


}



























