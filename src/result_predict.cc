#include "result_predict.h"

using namespace cv;
using namespace std;

int 
resultPredict(const Mat &_x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
 
    int nsamples = 1;
    vector<Mat> x;
    x.push_back(_x);
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

        double _factor = 0.0;
        double _max = max(tmpacti);
        double _min = min(tmpacti);
        if(fabs(_min) > fabs(_max)){
            _factor = _min / -5.0;
        }else{
            _factor = _max / 5.0;
        }
        if(_factor != 0) tmpacti = tmpacti.mul(1 / _factor);

        tmpacti = sigmoid(tmpacti);
        //tmpacti = Tanh(tmpacti);
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
    M.release();
    cpmap.clear();
    locmap.clear();
    hidden.clear();
    return result;
}

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

        double _factor = 0.0;
        double _max = max(tmpacti);
        double _min = min(tmpacti);
        if(fabs(_min) > fabs(_max)){
            _factor = _min / -5.0;
        }else{
            _factor = _max / 5.0;
        }
        if(_factor != 0) tmpacti = tmpacti.mul(1 / _factor);

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
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 100).
    // 
    int batchSize = 100;
    Mat result = Mat::zeros(1, testX.size(), CV_64FC1);
    vector<Mat> tmpBatch;
    int batch_amount = testX.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        cout<<"processing batch No. "<<i<<endl;
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(testX[i * batchSize + j]);
        }
        Mat resultBatch = resultPredict(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, 1);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
    }
    if(testX.size() % batchSize){
        cout<<"processing batch No. "<<batch_amount<<endl;
        for(int j = 0; j < testX.size() % batchSize; j++){
            tmpBatch.push_back(testX[batch_amount * batchSize + j]);
        }
        Mat resultBatch = resultPredict(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, testX.size() % batchSize, 1);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
    }

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
