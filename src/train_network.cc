#include "train_network.h"

using namespace cv;
using namespace std;

void
trainNetwork(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Fcl> &HiddenLayers, Smr &smr){

    if (is_gradient_checking){
        vector<Mat> tpx;
        tpx.push_back(x[0]);
        Rect roi = Rect(0, 0, 1, y.rows);
        Mat tpy;
        y(roi).copyTo(tpy);

        gradientChecking_ConvLayer(CLayers, HiddenLayers, smr, tpx, tpy);
        gradientChecking_FullConnectLayer(CLayers, HiddenLayers, smr, tpx, tpy);
        gradientChecking_SoftmaxLayer(CLayers, HiddenLayers, smr, tpx, tpy);
    }else{
    cout<<"****************************************************************************"<<endl
        <<"**                       TRAINING NETWORK......                             "<<endl
        <<"****************************************************************************"<<endl<<endl;

        // define the velocity vectors.
        Mat v_smr_W = Mat::zeros(smr.W.rows, smr.W.cols, CV_64FC1);
        Mat v_smr_b = Mat::zeros(smr.b.rows, smr.b.cols, CV_64FC1);
        vector<Mat> v_hl_W;
        vector<Mat> v_hl_b;
        for(int i = 0; i < HiddenLayers.size(); i ++){
            Mat tempW = Mat::zeros(HiddenLayers[i].W.rows, HiddenLayers[i].W.cols, CV_64FC1);
            Mat tempb = Mat::zeros(HiddenLayers[i].b.rows, HiddenLayers[i].b.cols, CV_64FC1);
            v_hl_W.push_back(tempW);
            v_hl_b.push_back(tempb);
        }
        vector<vector<Mat> > v_cvl_W;
        vector<vector<Scalar> > v_cvl_b;
        for(int cl = 0; cl < CLayers.size(); cl++){
            vector<Mat> tmpvecW;
            vector<Scalar> tmpvecb;
            if(convConfig[cl].is3chKernel){
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.rows, CLayers[cl].layer[i].W.cols, CV_64FC3);
                    Scalar tempb = Scalar(0.0, 0.0, 0.0);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                }
            }else{
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.rows, CLayers[cl].layer[i].W.cols, CV_64FC1);
                    Scalar tempb = Scalar(0.0);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                }
            }
            v_cvl_W.push_back(tmpvecW);
            v_cvl_b.push_back(tmpvecb);
        }
        mkdir(CLayers);
        double lr_w = lrate_w;
        double lr_b = lrate_b;
        double Momentum_w = 0.5;
        double Momentum_b = 0.5;
        for(int epo = 0; epo < training_epochs; epo++){
            for(int k = 0; k < iter_per_epo; k++){
                if(k > 30) {Momentum_w = 0.95; Momentum_b = 0.95;}
                int randomNum = ((long)rand() + (long)rand()) % (x.size() - batch_size - 1);
                vector<Mat> batchX;
                for(int i = 0; i < batch_size; i++){
                    batchX.push_back(x[i + randomNum]);
                }
                Rect roi = Rect(randomNum, 0, batch_size, y.rows);
                Mat batchY; 
                y(roi).copyTo(batchY);
                cout<<"epochs: "<<epo<<", learning step: "<<k;//<<endl;
                getNetworkCost(batchX, batchY, CLayers, HiddenLayers, smr);

                v_smr_W = v_smr_W * Momentum_w + lr_w * smr.Wgrad;
                v_smr_b = v_smr_b * Momentum_b + lr_b * smr.bgrad;
                smr.W -= v_smr_W;
                smr.b -= v_smr_b;
                for(int i = 0; i < HiddenLayers.size(); i++){
                    v_hl_W[i] = v_hl_W[i] * Momentum_w + lr_w * HiddenLayers[i].Wgrad;
                    v_hl_b[i] = v_hl_b[i] * Momentum_b + lr_b * HiddenLayers[i].bgrad;
                    HiddenLayers[i].W -= v_hl_W[i];
                    HiddenLayers[i].b -= v_hl_b[i];
                }
                for(int cl = 0; cl < CLayers.size(); cl++){
                    for(int i = 0; i < convConfig[cl].KernelAmount; i++){
                        v_cvl_W[cl][i] = v_cvl_W[cl][i] * Momentum_w + lr_w * CLayers[cl].layer[i].Wgrad;
                        v_cvl_b[cl][i] = v_cvl_b[cl][i] * Momentum_b + lr_b * CLayers[cl].layer[i].bgrad;
                        CLayers[cl].layer[i].W -= v_cvl_W[cl][i];
                        CLayers[cl].layer[i].b -= v_cvl_b[cl][i];
                    }
                }
                batchX.clear();
                batchY.release();
            }   
            save2txt(CLayers, epo);
            lr_w *= lrate_decay;
            lr_b *= lrate_decay;
        }
        v_smr_W.release();
        v_smr_b.release();
        v_hl_W.clear();
        v_hl_b.clear();
        v_cvl_W.clear();
        v_cvl_b.clear();
    }
}


