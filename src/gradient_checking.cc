#include "gradient_checking.h"

using namespace cv;
using namespace std;

void 
gradient_checking(const std::vector<Mat> &sampleX, const Mat &sampleY, std::vector<network_layer*> &flow, Mat &gradient, Mat* alt){
    Mat grad;
    gradient.copyTo(grad);
    double epsilon = 1e-5;
    if(alt -> channels() == 3){
        for(int i = 0; i < alt -> rows; i++){
            for(int j = 0; j < alt -> cols; j++){
                for(int ch = 0; ch < 3; ch++){
                    //cout<<"i = "<<i<<", j = "<<j<<", ch = "<<ch<<endl;
                    double memo = alt -> AT3D(i, j)[ch];
                    alt -> AT3D(i, j)[ch] = memo + epsilon;
                    forwardPass(sampleX, sampleY, flow);
                    double value1 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                    alt -> AT3D(i, j)[ch] = memo - epsilon;
                    forwardPass(sampleX, sampleY, flow);
                    double value2 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                    double tp = (value1 - value2) / (2 * epsilon);
                    if(tp == 0.0 && grad.AT3D(i, j)[ch] == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.AT3D(i, j)[ch]<<", "<<1<<endl;
                    else cout<<i<<", "<<j<<", "<<ch<<", "<<tp<<", "<<grad.AT3D(i, j)[ch]<<", "<<tp / grad.AT3D(i, j)[ch]<<endl;
                    alt -> AT3D(i, j)[ch] = memo;
                }
            }
        }
    }else{
        for(int i = 0; i < alt -> rows; i++){
            for(int j = 0; j < alt -> cols; j++){
                double memo = alt -> ATD(i, j);
                alt -> ATD(i, j) = memo + epsilon;
                forwardPass(sampleX, sampleY, flow);
                double value1 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                alt -> ATD(i, j) = memo - epsilon;
                forwardPass(sampleX, sampleY, flow);
                double value2 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                double tp = (value1 - value2) / (2 * epsilon);
                if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
                else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
                alt -> ATD(i, j) = memo;
            }
        }
    }
    grad.release();
}

void
gradientChecking_SoftmaxLayer(std::vector<network_layer*> &flow, const std::vector<Mat> &sampleX, const Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    
    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!!"<<endl;
    cout<<"################################################"<<endl;

    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "softmax"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;
            p = &(((softmax_layer*)flow[i]) -> w);
            gradient_checking(sampleX, sampleY, flow, ((softmax_layer*)flow[i]) -> wgrad, p);
        }
    }
}

void
gradientChecking_FullyConnectedLayer(std::vector<network_layer*> &flow, const std::vector<Mat> &sampleX, const Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)

    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test fully connected layer !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "fully_connected"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;
            p = &(((fully_connected_layer*)flow[i]) -> w);
            gradient_checking(sampleX, sampleY, flow, ((fully_connected_layer*)flow[i]) -> wgrad, p);
        }
    }

}

void
gradientChecking_ConvolutionalLayer(std::vector<network_layer*> &flow, const std::vector<Mat> &sampleX, const Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    
    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test convolutional layer !!!!"<<endl;
    cout<<"################################################"<<endl;

    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "convolutional"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;
            
            for(int j = 0; j < ((convolutional_layer*)flow[i]) -> kernels.size(); j++){
                cout<<"------ checking kernel number "<<j<<" ..."<<endl;
                p = &(((convolutional_layer*)flow[i]) -> kernels[j] -> w);
                gradient_checking(sampleX, sampleY, flow, ((convolutional_layer*)flow[i]) -> kernels[j] -> wgrad, p);
            }          
            cout<<"-------------------------------------- checking combine feature map weight"<<endl;
            p = &(((convolutional_layer*)flow[i]) -> combine_weight);
            gradient_checking(sampleX, sampleY, flow, ((convolutional_layer*)flow[i]) -> combine_weight_grad, p);
        }
    }
}

void 
gradient_checking_network_layers(std::vector<network_layer*> &flow, const std::vector<Mat> &sampleX, const Mat &sampleY){

    // delete dropout layer when doing gradient checking
    std::vector<network_layer*> tmpflow(flow);
    int i = 0;
    while(true){
        if(i >= tmpflow.size()) break;
        if(tmpflow[i] -> layer_type == "dropout"){
            tmpflow.erase(tmpflow.begin() + i);
        }else ++i;
    }
    gradientChecking_ConvolutionalLayer(tmpflow, sampleX, sampleY);
    //gradientChecking_FullyConnectedLayer(tmpflow, sampleX, sampleY);
    //gradientChecking_SoftmaxLayer(tmpflow, sampleX, sampleY);

}






