#include "result_predict.h"

using namespace cv;
using namespace std;

int 
resultPredict(const Mat &_x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
 
    int nsamples = 1;
    vector<Mat> x;
    x.push_back(_x);
    // Conv & Pooling
    vector<vector<Mat> > conved;
    convAndPooling(x, CLayers, conved);
    splitChannels(conved);
    Mat convolvedX = concatenateMat(conved);

    // full connected layers
    vector<Mat> hidden;
    hidden.push_back(convolvedX);
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
//        tmpacti = sigmoid(tmpacti);
        tmpacti = ReLU(tmpacti);
        if(fcConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(fcConfig[i - 1].DropoutRate);
        hidden.push_back(tmpacti);
    }
    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);

    int result = 0;
    double minValue, maxValue;
    Point minLoc, maxLoc;
    minMaxLoc(M(Rect(0, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
    result = (int) maxLoc.y;
    // destructor
    for(int i = 0; i < conved.size(); i++){
        conved[i].clear();
    }
    conved.clear();
    M.release();
    hidden.clear();
    convolvedX.release();
    return result;
}

Mat 
resultPredict(const vector<Mat> &x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
 
    int nsamples = x.size();
    // Conv & Pooling
    vector<vector<Mat> > conved;
    convAndPooling(x, CLayers, conved);
    splitChannels(conved);
    Mat convolvedX = concatenateMat(conved);

    // full connected layers
    vector<Mat> hidden;
    hidden.push_back(convolvedX);
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
//        tmpacti = sigmoid(tmpacti);
        tmpacti = ReLU(tmpacti);
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
    for(int i = 0; i < conved.size(); i++){
        conved[i].clear();
    }
    conved.clear();
    M.release();
    hidden.clear();
    convolvedX.release();
    return result;
}


void 
testNetwork(const vector<Mat> &testX, const Mat &testY, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){   
    // fixed the result predict method, avoid using hash map, so 
    // it's ok to do all of them in one call
    Mat result = resultPredict(testX, CLayers, hLayers, smr);
    Mat err;
    testY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.ATD(0, i) != 0) --correct;
    }
    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    result.release();
    err.release();
}
