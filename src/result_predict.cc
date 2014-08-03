#include "result_predict.h"

using namespace cv;
using namespace std;


Mat resultProdict(vector<Mat> &x, vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr, double lambda){
 
    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    unordered_map<string, vector<vector<Point> > > locmap;
    convAndPooling(x, CLayers, cpmap, locmap, true);

    vector<Mat> P;
    vector<string> vecstr = getLayerKey(nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i = 0; i<vecstr.size(); i++){
        P.push_back(cpmap[vecstr[i]]);
    }
    Mat convolvedX = concatenateMat(P, nsamples);
    P.clear();

    // full connected layers
    vector<Mat> hidden;
    hidden.push_back(convolvedX);
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
        tmpacti = sigmoid(tmpacti);
        //tmpacti = Tanh(tmpacti);
        if(DROPOUT) tmpacti = tmpacti.mul(fcConfig[i - 1].DropoutRate);
        hidden.push_back(tmpacti);
    }

    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M -= repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);

    Mat logP;
    log(p, logP);

    Mat result = Mat::ones(1, logP.cols, CV_64FC1);
    for(int i=0; i<logP.cols; i++){
        double maxele = logP.ATD(0, i);
        int which = 0;
        for(int j=1; j<logP.rows; j++){
            if(logP.ATD(j, i) > maxele){
                maxele = logP.ATD(j, i);
                which = j;
            }
        }
        result.ATD(0, i) = which;
    }
    // destructor
    p.release();
    M.release();
    tmp.release();
    logP.release();
    cpmap.clear();
    locmap.clear();
    hidden.clear();
    return result;
}














