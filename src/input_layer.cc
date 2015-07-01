#include "layer_bank.h"

using namespace std;

input_layer::input_layer(){}
input_layer::~input_layer(){
    
}

void input_layer::init_config(string namestr, int batchsize, string outputformat){
    layer_type = "input";
    layer_name = namestr;
    batch_size = batchsize;
    output_format = outputformat;
    label = Mat::zeros(1, batch_size, CV_64FC1);
}

void input_layer::forwardPass(int nsamples, const vector<Mat>& input_data, const Mat& input_label){
    getSample(input_data, output_vector, input_label, label);
    //cout<<"----------------"<<endl<<label<<endl;
}

void input_layer::forwardPassTest(int nsamples, const vector<Mat>& input_data, const Mat& input_label){
    
    output_vector.resize(input_data.size());
    for(int i = 0; i < output_vector.size(); i++){
        output_vector[i].resize(1);
    }
    for(int i = 0; i < input_data.size(); i++){
        input_data[i].copyTo(output_vector[i][0]);
    }
    input_label.copyTo(label);
}

void input_layer::getSample(const vector<Mat>& src1, vector<vector<Mat> >& dst1, const Mat& src2, Mat& dst2){
    
    dst1.clear();
    if(is_gradient_checking){
        for(int i = 0; i < batch_size; i++){
            vector<Mat> tmp;
            tmp.push_back(src1[i]);
            dst1.push_back(tmp);
            dst2.ATD(0, i) = src2.ATD(0, i);
        }
        return;
    }
    if(src1.size() < batch_size){
        for(int i = 0; i < src1.size(); i++){
            vector<Mat> tmp;
            tmp.push_back(src1[i]);
            dst1.push_back(tmp);
        }
        Rect roi = Rect(0, 0, src2.cols, 1);
        src2(roi).copyTo(dst2);
        return;
    }
    vector<int> sample_vec;
    for(int i = 0; i < src1.size(); i++){
        sample_vec.push_back(i);
    }
    random_shuffle(sample_vec.begin(), sample_vec.end());
    for(int i = 0; i < batch_size; i++){
        vector<Mat> tmp;
        tmp.push_back(src1[sample_vec[i]]);
        dst1.push_back(tmp);
        dst2.ATD(0, i) = src2.ATD(0, sample_vec[i]);
    }
}

void input_layer::backwardPass(){
    ;
}
































