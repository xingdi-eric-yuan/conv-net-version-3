#include "result_predict.h"

using namespace cv;
using namespace std;


Mat 
resultPredict(const vector<Mat> &x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
 
    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    unordered_map<string, vector<vector<Point> > > locmap;
    convAndPooling(x, CLayers, cpmap, locmap, true);
    vector<Mat> P;
    vector<string> vecstr = getLayerKey(nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i = 0; i<vecstr.size(); i++){
        P.push_back(cpmap.at(vecstr[i]));
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
        if(fcConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(fcConfig[i - 1].DropoutRate);
        hidden.push_back(tmpacti);
    }
    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);
    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int) maxLoc.y;
    }
    // destructor
    M.release();
    cpmap.clear();
    locmap.clear();
    hidden.clear();
    return result;
}

void 
testNetwork(const vector<Mat> &testX, const Mat &testY, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
    // Test use test set
    Mat result = resultPredict(testX, CLayers, hLayers, smr);
    Mat err(testY);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.ATD(0, i) != 0) --correct;
    }
    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    result.release();
    err.release();
}











