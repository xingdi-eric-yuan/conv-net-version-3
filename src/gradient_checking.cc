#include "gradient_checking.h"

using namespace cv;
using namespace std;

void
gradientChecking_ConvLayer(vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr, vector<Mat> &x, Mat &y, double lambda){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    DROPOUT = false;
    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
    int a = 0;
    int b = 0;
    Mat grad(CLayers[a].layer[b].Wgrad);
    cout<<"test network !!!!"<<endl;
    double epsilon = 1e-4;
    if(convConfig[a].is3chKernel){
        for(int i=0; i<CLayers[a].layer[b].W.rows; i++){
            for(int j=0; j<CLayers[a].layer[b].W.cols; j++){
                for(int ch = 0; ch < CLayers[a].layer[b].W.channels(); ch++){
                    double memo = CLayers[a].layer[b].W.AT3D(i, j)[ch];
                    CLayers[a].layer[b].W.AT3D(i, j)[ch] = memo + epsilon;
                    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
                    double value1 = smr.cost;
                    CLayers[a].layer[b].W.AT3D(i, j)[ch] = memo - epsilon;
                    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
                    double value2 = smr.cost;
                    double tp = (value1 - value2) / (2 * epsilon);
                    cout<<i<<", "<<j<<", "<<ch<<", "<<tp<<", "<<grad.AT3D(i, j)[ch]<<", "<<grad.AT3D(i, j)[ch] / tp<<endl;
                    CLayers[a].layer[b].W.AT3D(i, j)[ch] = memo;
                }
            }
        }
    }else{
        for(int i=0; i<CLayers[a].layer[b].W.rows; i++){
            for(int j=0; j<CLayers[a].layer[b].W.cols; j++){
                double memo = CLayers[a].layer[b].W.ATD(i, j);
                CLayers[a].layer[b].W.ATD(i, j) = memo + epsilon;
                getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
                double value1 = smr.cost;
                CLayers[a].layer[b].W.ATD(i, j) = memo - epsilon;
                getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
                double value2 = smr.cost;
                double tp = (value1 - value2) / (2 * epsilon);
                cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<grad.ATD(i, j) / tp<<endl;
                CLayers[a].layer[b].W.ATD(i, j) = memo;
            }
        }
    }
    grad.release();
}

void
gradientChecking_FullConnectLayer(vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr, vector<Mat> &x, Mat &y, double lambda){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    DROPOUT = false;
    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
    int a = 0;
    int b = 0;
    Mat grad(hLayers[a].Wgrad);
    cout<<"test network !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<hLayers[a].W.rows; i++){
        for(int j=0; j<hLayers[a].W.cols; j++){
            double memo = hLayers[a].W.ATD(i, j);
            hLayers[a].W.ATD(i, j) = memo + epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value1 = smr.cost;
            hLayers[a].W.ATD(i, j) = memo - epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<grad.ATD(i, j) / tp<<endl;
            hLayers[a].W.ATD(i, j) = memo;
        }
    }
    grad.release();
}

void
gradientChecking_SoftmaxLayer(vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr, vector<Mat> &x, Mat &y, double lambda){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    DROPOUT = false;
    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
    Mat grad(smr.Wgrad);
    cout<<"test network !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<smr.W.rows; i++){
        for(int j=0; j<smr.W.cols; j++){
            double memo = smr.W.ATD(i, j);
            smr.W.ATD(i, j) = memo + epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value1 = smr.cost;
            smr.W.ATD(i, j) = memo - epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<grad.ATD(i, j) / tp<<endl;
            smr.W.ATD(i, j) = memo;
        }
    }
    grad.release();
}

