#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

class convolutional_kernel{
public:
    convolutional_kernel();
    ~convolutional_kernel();
    void init_config(int, double);

    Mat w;
    Scalar b;
    Mat wgrad;
    Scalar bgrad;
    Mat wd2;
    Scalar bd2;
    int kernel_size;
    double weight_decay;

};

class network_layer{
public:
    network_layer();
    virtual ~network_layer();

    Mat output_matrix;
    std::vector<std::vector<Mat> > output_vector;
    Mat delta_matrix;
    std::vector<std::vector<Mat> > delta_vector;
    Mat d2_matrix;
    std::vector<std::vector<Mat> > d2_vector;

    std::string layer_name;
    std::string layer_type;
    std::string output_format;
    /// layer type:
    // input
    // convolutional
    // full_connected
    // pooling
    // softmax
    // local_response_normalization
    // dropout
    // non_linearity
};

class input_layer : public network_layer{
public:
    input_layer();
    ~input_layer();
    void init_config(string, int, string);
    void forwardPass(int, const vector<Mat>&, const Mat&);
    void forwardPassTest(int, const vector<Mat>&, const Mat&);
    void getSample(const std::vector<Mat>&, std::vector<vector<Mat> >&, const Mat&, Mat&);
    void backwardPass();
    Mat label;
    int batch_size;

};

class convolutional_layer : public network_layer{
public:
    convolutional_layer();
    ~convolutional_layer();
    void init_config(string, int, int, int, int, int, double, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    std::vector<convolutional_kernel*> kernels;
    Mat combine_weight;
    Mat combine_weight_grad;
    Mat combine_weight_d2;
    int padding;
    int stride;
    int combine_feature_map;

    // updater    
    void setMomentum();
    void update(int);
    double momentum_derivative;
    double momentum_second_derivative;
    int iter;
    double mu;
    std::vector<Mat> velocity_w;
    std::vector<Scalar> velocity_b;
    std::vector<Mat> second_derivative_w;
    std::vector<Scalar> second_derivative_b;
    Mat velocity_combine_weight;
    Mat second_derivative_combine_weight;
    Mat learning_rate_w;
    Scalar learning_rate_b;
};

class pooling_layer : public network_layer{
public:
    pooling_layer();
    ~pooling_layer();

    void init_config(string, int, string, int, int);
    void init_config(string, int, string, int);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    int stride;
    int window_size;
    int method;
    bool overlap;
    std::vector<std::vector<std::vector<std::vector<Point> > > > location;
};

class fully_connected_layer : public network_layer{
public:
    fully_connected_layer();
    ~fully_connected_layer();
    void init_config(string, int, double, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    Mat w;
    Mat b;
    Mat wgrad;
    Mat bgrad;
    Mat wd2;
    Mat bd2;

    int size;
    double weight_decay;

    // updater
    
    void setMomentum();
    void update(int);
    double momentum_derivative;
    double momentum_second_derivative;
    int iter;
    double mu;
    Mat velocity_w;
    Mat velocity_b;
    Mat second_derivative_w;
    Mat second_derivative_b;
    Mat learning_rate;
};

class softmax_layer : public network_layer{
public:
    softmax_layer();
    ~softmax_layer();
    void init_config(string, int, double, string);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void init_weight(network_layer*);
    void backwardPass(int, network_layer*, Mat&);

    Mat w;
    Mat b;
    Mat wgrad;
    Mat bgrad;
    Mat wd2;
    Mat bd2;
    double network_cost;
    int output_size;
    double weight_decay;
    
    // updater
    void setMomentum();
    void update(int);

    double momentum_derivative;
    double momentum_second_derivative;
    int iter;
    double mu;
    Mat velocity_w;
    Mat velocity_b;
    Mat second_derivative_w;
    Mat second_derivative_b;
    Mat learning_rate;
};

class local_response_normalization_layer : public network_layer{
public:
    
    local_response_normalization_layer();
    ~local_response_normalization_layer();
    void init_config(string, string, double, double, double, int);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);
    Mat local_response_normalization(std::vector<Mat>&, int);
    Mat dlocal_response_normalization(std::vector<Mat>&, int);

    double alpha;
    double beta;
    double k;
    int n;
};

class dropout_layer : public network_layer{
public:
    dropout_layer();
    ~dropout_layer();
    void init_config(string, string, double);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    double dropout_rate;
    Mat bernoulli_matrix;
    std::vector<std::vector<Mat> > bernoulli_vector;
};

class non_linearity_layer : public network_layer{
public:
    non_linearity_layer();
    ~non_linearity_layer();
    void init_config(string, int, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    int method;
};

class branch_layer : public network_layer{
public:
    branch_layer();
    ~branch_layer();
    void init_config(string, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

};

class combine_layer : public network_layer{
public:
    combine_layer();
    ~combine_layer();
    void init_config(string, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

};















