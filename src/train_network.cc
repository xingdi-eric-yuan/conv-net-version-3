#include "train_network.h"

using namespace cv;
using namespace std;

void
trainNetwork(const vector<Mat> &x, const Mat &y, vector<Cvl> &CLayers, vector<Fcl> &HiddenLayers, Smr &smr, const vector<Mat> &tx, const Mat &ty){

    if (is_gradient_checking){
        vector<Mat> tpx;
        Mat tpy;
        getSample(x, &tpx, y, &tpy, 1, SAMPLE_COLS);
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
        Mat smrWd2 = Mat::zeros(smr.W.size(), CV_64FC1);
        Mat smrbd2 = Mat::zeros(smr.b.size(), CV_64FC1);

        vector<Mat> v_hl_W;
        vector<Mat> v_hl_b;
        vector<Mat> hlWd2;
        vector<Mat> hlbd2;
        for(int i = 0; i < HiddenLayers.size(); i ++){
            Mat tempW = Mat::zeros(HiddenLayers[i].W.size(), CV_64FC1);
            Mat tempb = Mat::zeros(HiddenLayers[i].b.size(), CV_64FC1);
            Mat tempWd2 = Mat::zeros(tempW.size(), CV_64FC1);
            Mat tempbd2 = Mat::zeros(tempb.size(), CV_64FC1);
            v_hl_W.push_back(tempW);
            v_hl_b.push_back(tempb);
            hlWd2.push_back(tempWd2);
            hlbd2.push_back(tempbd2);
        }
        
        vector<vector<Mat> > v_cvl_W;
        vector<vector<Scalar> > v_cvl_b;
        vector<vector<Mat> > cvlWd2;
        vector<vector<Scalar> > cvlbd2;
        for(int cl = 0; cl < CLayers.size(); cl++){
            vector<Mat> tmpvecW;
            vector<Scalar> tmpvecb;
            vector<Mat> tmpvecWd2;
            vector<Scalar> tmpvecbd2;
            if(convConfig[cl].is3chKernel){
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.size(), CV_64FC3);
                    Scalar tempb = Scalar(0.0, 0.0, 0.0);
                    Mat tempWd2 = Mat::zeros(tempW.size(), CV_64FC3);
                    Scalar tempbd2 = Scalar(0.0, 0.0, 0.0);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                    tmpvecWd2.push_back(tempWd2);
                    tmpvecbd2.push_back(tempbd2);
                }
            }else{
                for(int i = 0; i < convConfig[cl].KernelAmount; i ++){
                    Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.size(), CV_64FC1);
                    Scalar tempb = Scalar(0.0);
                    Mat tempWd2 = Mat::zeros(tempW.size(), CV_64FC1);
                    Scalar tempbd2 = Scalar(0.0);
                    tmpvecW.push_back(tempW);
                    tmpvecb.push_back(tempb);
                    tmpvecWd2.push_back(tempWd2);
                    tmpvecbd2.push_back(tempbd2);
                }
            }
            v_cvl_W.push_back(tmpvecW);
            v_cvl_b.push_back(tmpvecb);
            cvlWd2.push_back(tmpvecWd2);
            cvlbd2.push_back(tmpvecbd2);
        }
        double Momentum_w = 0.5;
        double Momentum_b = 0.5;
        double Momentum_d2 = 0.5;
        Mat lr_w;
        Mat lr_b;
        Scalar lr_b_s;
        //double lr_b = lr_b_global;
        double mu = 1e-2;
        int k = 0;
        for(int epo = 1; epo <= training_epochs; epo++){
            for(; k <= iter_per_epo * epo; k++){
                log_iter = k;
                string path = "log/iter_" + to_string((long long)log_iter);
                $$LOG 
#ifdef _WIN32
				string p = std::string("md ") + path;
				replace_if(p.begin(), p.end(), bind2nd(std::equal_to<char>(),'/'), '\\');
				std::cout<<p<<endl;

				system(p.c_str());
#else
					mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 
#endif
					
				$$_LOG
                if(k > 30) {Momentum_w = 0.95; Momentum_b = 0.95; Momentum_d2 = 0.90;}
                vector<Mat> batchX;
                Mat batchY; 
                getSample(x, &batchX, y, &batchY, batch_size, SAMPLE_COLS);
                //if(k == 20) getNetworkLearningRate(batchX, batchY, CLayers, HiddenLayers, smr);     
                cout<<"epoch: "<<epo<<", iter: "<<k;//<<endl;           
                getNetworkCost(batchX, batchY, CLayers, HiddenLayers, smr);
                // softmax update
                smrWd2 = Momentum_d2 * smrWd2 + (1.0 - Momentum_d2) * smr.Wd2;
                smrbd2 = Momentum_d2 * smrbd2 + (1.0 - Momentum_d2) * smr.bd2;
                lr_w = smr.lr_w / (smrWd2 + mu);
                lr_b = smr.lr_b / (smrbd2 + mu);
                v_smr_W = v_smr_W * Momentum_w + (1.0 - Momentum_w) * smr.Wgrad.mul(lr_w);
                v_smr_b = v_smr_b * Momentum_b + (1.0 - Momentum_b) * smr.bgrad.mul(lr_b);
                smr.W -= v_smr_W;
                smr.b -= v_smr_b;
                // full-connected layer update
                for(int i = 0; i < HiddenLayers.size(); i++){
                    hlWd2[i] = Momentum_d2 * hlWd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].Wd2;
                    hlbd2[i] = Momentum_d2 * hlbd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].bd2;
                    lr_w = HiddenLayers[i].lr_w / (hlWd2[i] + mu);
                    lr_b = HiddenLayers[i].lr_b / (hlbd2[i] + mu);
                    v_hl_W[i] = v_hl_W[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].Wgrad.mul(lr_w);
                    v_hl_b[i] = v_hl_b[i] * Momentum_b + (1.0 - Momentum_b) * HiddenLayers[i].bgrad.mul(lr_b);
                    HiddenLayers[i].W -= v_hl_W[i];
                    HiddenLayers[i].b -= v_hl_b[i];
                }
                // convolutional layer update
                for(int cl = 0; cl < CLayers.size(); cl++){
                    for(int i = 0; i < convConfig[cl].KernelAmount; i++){
                        cvlWd2[cl][i] = Momentum_d2 * cvlWd2[cl][i] + (1.0 - Momentum_d2) * CLayers[cl].layer[i].Wd2;
                        cvlbd2[cl][i] = Momentum_d2 * cvlbd2[cl][i] + (1.0 - Momentum_d2) * CLayers[cl].layer[i].bd2;
                        lr_w = CLayers[cl].layer[i].lr_w / (cvlWd2[cl][i] + mu);
                        if(convConfig[cl].is3chKernel) lr_b_s = CLayers[cl].layer[i].lr_b / (cvlbd2[cl][i] + Scalar(mu, mu, mu));
                        else lr_b_s = CLayers[cl].layer[i].lr_b / (cvlbd2[cl][i] + Scalar(mu));
                        v_cvl_W[cl][i] = v_cvl_W[cl][i] * Momentum_w + (1.0 - Momentum_w) * CLayers[cl].layer[i].Wgrad.mul(lr_w);                        
                        v_cvl_b[cl][i] = v_cvl_b[cl][i] * Momentum_b + (1.0 - Momentum_b) * CLayers[cl].layer[i].bgrad.mul(lr_b_s);
                        CLayers[cl].layer[i].W -= v_cvl_W[cl][i];
                        CLayers[cl].layer[i].b -= v_cvl_b[cl][i];
                    }
                }
                batchX.clear();
                batchY.release();
                $$LOG saveConvKernel(CLayers, path); $$_LOG
            } 
            if(! is_gradient_checking){
                cout<<"Test training data: ";
                testNetwork(x, y, CLayers, HiddenLayers, smr);
                cout<<"Test testing data: ";
                testNetwork(tx, ty, CLayers, HiddenLayers, smr);
            }
        }

        v_smr_W.release();
        v_smr_b.release();
        v_hl_W.clear();
        v_hl_b.clear();
        v_cvl_W.clear();
        v_cvl_b.clear();
        smrWd2.release();
        hlWd2.clear();
        cvlWd2.clear();
    }
}


