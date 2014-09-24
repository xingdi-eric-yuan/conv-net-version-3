#include "train_network.h"

using namespace cv;
using namespace std;

void
trainNetwork(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Fcl> &HiddenLayers, Smr &smr){

    if (is_gradient_checking){
        vector<Mat> tpx;
        Mat tpy;
        getSample(x, tpx, y, tpy, 1, SAMPLE_COLS);
        gradientChecking_ConvLayer(CLayers, HiddenLayers, smr, tpx, tpy);
        gradientChecking_FullConnectLayer(CLayers, HiddenLayers, smr, tpx, tpy);
        gradientChecking_SoftmaxLayer(CLayers, HiddenLayers, smr, tpx, tpy);
    }else{
    cout<<"****************************************************************************"<<endl
        <<"**                       TRAINING NETWORK......                             "<<endl
        <<"****************************************************************************"<<endl<<endl;

        // define the velocity vectors.
        Mat v_smr_W = Mat::zeros(smr.W.size(), CV_64FC1);
        Mat v_smr_b = Mat::zeros(smr.b.size(), CV_64FC1);
        Mat smrd2 = Mat::zeros(v_smr_W.size(), CV_64FC1);

        vector<Mat> v_hl_W;
        vector<Mat> v_hl_b;
        vector<Mat> hld2;
        for(int i = 0; i < HiddenLayers.size(); i ++){
            Mat tempW = Mat::zeros(HiddenLayers[i].W.size(), CV_64FC1);
            Mat tempb = Mat::zeros(HiddenLayers[i].b.size(), CV_64FC1);
            Mat tempd2 = Mat::zeros(tempW.size(), CV_64FC1);
            v_hl_W.push_back(tempW);
            v_hl_b.push_back(tempb);
            hld2.push_back(tempd2);
        }
        
        vector<vector<Mat> > v_cvl_W;
        vector<vector<Scalar> > v_cvl_b;
        vector<vector<Mat> > cvld2;
        for(int cl = 0; cl < CLayers.size(); cl++){
            vector<Mat> tmpvecW;
            vector<Scalar> tmpvecb;
            vector<Mat> tmpvecd2;
            if(convConfig[cl].is3chKernel){
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.size(), CV_64FC3);
                    Scalar tempb = Scalar(0.0, 0.0, 0.0);
                    Mat tempd2 = Mat::zeros(tempW.size(), CV_64FC3);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                    tmpvecd2.push_back(tempd2);
                }
            }else{
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.size(), CV_64FC1);
                    Scalar tempb = Scalar(0.0);
                    Mat tempd2 = Mat::zeros(tempW.size(), CV_64FC1);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                    tmpvecd2.push_back(tempd2);
                }
            }
            v_cvl_W.push_back(tmpvecW);
            v_cvl_b.push_back(tmpvecb);
            cvld2.push_back(tmpvecd2);
        }
        double Momentum_w = 0.5;
        double Momentum_b = 0.5;
        double Momentum_d2 = 0.5;
        double lr_w_global = lrate_w;
        double lr_b_global = lrate_b;
        Mat lr_w;
        double lr_b = lr_b_global;
        double mu = 0.3;
        for(int k = 0; k <= iter_per_epo; k++){
            log_iter = k;
            string path = "log/iter_" + to_string(log_iter);
            $$LOG mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); $$_LOG

            if(k > 30) {Momentum_w = 0.95; Momentum_b = 0.95; Momentum_d2 = 0.9;}
            vector<Mat> batchX;
            Mat batchY; 
            getSample(x, batchX, y, batchY, batch_size, SAMPLE_COLS);
            //if(k == 20) getNetworkLearningRate(batchX, batchY, CLayers, HiddenLayers, smr);     
            cout<<"iter: "<<k<<", learning step: "<<k;//<<endl;           
            getNetworkCost(batchX, batchY, CLayers, HiddenLayers, smr);
            // softmax update
            smrd2 = Momentum_d2 * smrd2 + (1.0 - Momentum_d2) * smr.d2;
            lr_w = lr_w_global / (smrd2 + mu);
            v_smr_W = v_smr_W * Momentum_w + smr.Wgrad.mul(lr_w);
            v_smr_b = v_smr_b * Momentum_b + smr.bgrad.mul(lr_b);
            smr.W -= v_smr_W;
            smr.b -= v_smr_b;
            // full-connected layer update
            for(int i = 0; i < HiddenLayers.size(); i++){
                hld2[i] = Momentum_d2 * hld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].d2;
                lr_w = lr_w_global / (hld2[i] + mu);
                v_hl_W[i] = v_hl_W[i] * Momentum_w + HiddenLayers[i].Wgrad.mul(lr_w);
                v_hl_b[i] = v_hl_b[i] * Momentum_b + HiddenLayers[i].bgrad.mul(lr_b);
                HiddenLayers[i].W -= v_hl_W[i];
                HiddenLayers[i].b -= v_hl_b[i];
            }
            // convolutional layer update
            for(int cl = 0; cl < CLayers.size(); cl++){
                for(int i = 0; i < convConfig[cl].KernelAmount; i++){
                    cvld2[cl][i] = Momentum_d2 * cvld2[cl][i] + (1.0 - Momentum_d2) * CLayers[cl].layer[i].d2;
                    lr_w = lr_w_global / (cvld2[cl][i] + mu);
                    v_cvl_W[cl][i] = v_cvl_W[cl][i] * Momentum_w + CLayers[cl].layer[i].Wgrad.mul(lr_w);                        
                    v_cvl_b[cl][i] = v_cvl_b[cl][i] * Momentum_b + CLayers[cl].layer[i].bgrad.mul(lr_b);
                    CLayers[cl].layer[i].W -= v_cvl_W[cl][i];
                    CLayers[cl].layer[i].b -= v_cvl_b[cl][i];
                }
            }
            batchX.clear();
            batchY.release();
            $$LOG saveConvKernel(CLayers, path); $$_LOG
        }   
        v_smr_W.release();
        v_smr_b.release();
        v_hl_W.clear();
        v_hl_b.clear();
        v_cvl_W.clear();
        v_cvl_b.clear();
        smrd2.release();
        hld2.clear();
        cvld2.clear();
    }
}


