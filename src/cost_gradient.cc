#include "cost_gradient.h"

using namespace cv;
using namespace std;

void
getNetworkCost(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr){

    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    unordered_map<string, vector<vector<Point> > > locmap;
    convAndPooling(x, CLayers, cpmap, locmap, false);
    vector<Mat> P;
    vector<string> vecstr = getLayerKey(nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i = 0; i<vecstr.size(); i++){
        P.push_back(cpmap.at(vecstr[i]));
    }
    Mat convolvedX = concatenateMat(P, nsamples);
    P.clear();

    // full connected layers
    vector<Mat> hidden;
    vector<Mat> acti;
    vector<double> factor;
    hidden.push_back(convolvedX);
    acti.push_back(convolvedX);
    vector<Mat> bernoulli;
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
        //tmpacti = Tanh(tmpacti);
        double _factor = 0.0;
        double _max = max(tmpacti);
        double _min = min(tmpacti);
        if(fabs(_min) > fabs(_max)){
            _factor = _min / -5.0;
        }else{
            _factor = _max / 5.0;
        }
        if(_factor != 0) tmpacti = tmpacti.mul(1 / _factor);
        factor.push_back(_factor);
        tmpacti = sigmoid(tmpacti);
        if(fcConfig[i - 1].DropoutRate < 1.0){
            Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, fcConfig[i - 1].DropoutRate);
            hidden.push_back(tmpacti.mul(bnl));
            bernoulli.push_back(bnl);
        }else hidden.push_back(tmpacti);
        acti.push_back(tmpacti);
    }

    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
    M = exp(M);
    Mat p = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));

    Mat groundTruth = Mat::zeros(softmaxConfig.NumClasses, nsamples, CV_64FC1);
    for(int i = 0; i < nsamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }
    double J1 = - sum1(groundTruth.mul(log(p))) / nsamples;
    double J2 = sum1(pow(smr.W, 2.0)) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += sum1(pow(hLayers[hl].W, 2.0)) * fcConfig[hl].WeightDecay / 2;
    }
    for(int cl = 0; cl < CLayers.size(); cl++){
        for(int i = 0; i < convConfig[cl].KernelAmount; i++){
            if(convConfig[cl].is3chKernel)
                J4 += sum1(pow(CLayers[cl].layer[i].W, 2.0)) * convConfig[cl].WeightDecay / 2;
            else 
                J4 += 3 * sum1(pow(CLayers[cl].layer[i].W, 2.0)) * convConfig[cl].WeightDecay / 2;
        }
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;
    // bp - softmax
    smr.Wgrad =  - (groundTruth - p) * hidden[hidden.size() - 1].t() / nsamples + softmaxConfig.WeightDecay * smr.W;
    smr.bgrad = - reduce((groundTruth - p), 1, CV_REDUCE_SUM) / nsamples;

    // bp - full connected
    vector<Mat> delta(hidden.size());
    delta[delta.size() -1] = -smr.W.t() * (groundTruth - p);
    delta[delta.size() -1] = delta[delta.size() -1].mul(dsigmoid(acti[acti.size() - 1]));
    if(factor[factor.size() - 1] != 0) delta[delta.size() -1] = delta[delta.size() -1].mul(1 / factor[factor.size() - 1]);
    //delta[delta.size() -1] = delta[delta.size() -1].mul(dTanh(acti[acti.size() - 1]));
    if(fcConfig[fcConfig.size() - 1].DropoutRate < 1.0) delta[delta.size() - 1] = delta[delta.size() -1].mul(bernoulli[bernoulli.size() - 1]);
    for(int i = delta.size() - 2; i >= 0; i--){
        delta[i] = hLayers[i].W.t() * delta[i + 1];
        if(i > 0){
            delta[i] = delta[i].mul(dsigmoid(acti[i]));
            if(factor[i] != 0) delta[i] = delta[i].mul(1 / factor[i]);
            //delta[i] = delta[i].mul(dTanh(acti[i]));
            if(fcConfig[i - 1].DropoutRate < 1.0) delta[i] = delta[i].mul(bernoulli[i - 1]);
        } 
    }
    for(int i = fcConfig.size() - 1; i >= 0; i--){
        hLayers[i].Wgrad = delta[i + 1] * (hidden[i]).t();
        hLayers[i].Wgrad = hLayers[i].Wgrad / nsamples + fcConfig[i].WeightDecay * hLayers[i].W;
        hLayers[i].bgrad = reduce(delta[i + 1], 1, CV_REDUCE_SUM) / nsamples;
    }
    //bp - Conv layer
    hashDelta(delta[0], cpmap, CLayers);
    for(int cl = CLayers.size() - 1; cl >= 0; cl --){
        int pDim = convConfig[cl].PoolingDim;
        vector<string> deltaKey = getLayerKey(nsamples, cl, KEY_DELTA);
        for(int k = 0; k < deltaKey.size(); k ++){
            string locstr = deltaKey[k].substr(0, deltaKey[k].length() - 1);
            Mat upDelta = UnPooling(cpmap.at(deltaKey[k]), pDim, pDim, pooling_method, locmap.at(locstr));
            string upDstr = locstr + "UD";
            cpmap[upDstr] = upDelta;
        }
        for(int k = 0; k < deltaKey.size(); k ++){
            string locstr = deltaKey[k].substr(0, deltaKey[k].length() - 1);
            string convstr = deltaKey[k].substr(0, deltaKey[k].length() - 2);
            // Local response normalization 
            string upDstr = locstr + "UD";
            if(convConfig[cl].useLRN){
                Mat dLRN = dlocalResponseNorm(cpmap, upDstr); 
                dLRN = dLRN.mul(dnonLinearityC3(cpmap[convstr]));
                cpmap[upDstr] = dLRN;
            }else{
                Mat upDelta;
                cpmap.at(upDstr).copyTo(upDelta);
                upDelta = upDelta.mul(dnonLinearityC3(cpmap.at(convstr)));
                cpmap[upDstr] = upDelta;
            }
        }
        if(cl > 0){
            for(int k = 0; k < convConfig[cl - 1].KernelAmount; k ++){
                vector<string> prev = getSpecKeys(nsamples, cl, cl - 1, k, KEY_UP_DELTA);
                for(int i = 0; i < prev.size(); i++){
                    string strd = getPreviousLayerKey(prev[i], KEY_DELTA);
                    unordered_map<string, Mat>::iterator got = cpmap.find(strd);
                    if(got == cpmap.end()){
                        string psize = getPreviousLayerKey(prev[i], KEY_POOL);
                        Mat zero = Mat::zeros(cpmap.at(psize).rows, cpmap.at(psize).cols, CV_64FC3);
                        cpmap[strd] = zero;
                    }
                    int currentKernel = getCurrentKernelNum(prev[i]);
                     cpmap.at(strd) += convCalc(cpmap.at(prev[i]), CLayers[cl].layer[currentKernel].W, CONV_FULL);
                }
            }
        }
        for(int j = 0; j < convConfig[cl].KernelAmount; j ++){
            Mat tpgradW = Mat::zeros(convConfig[cl].KernelSize, convConfig[cl].KernelSize, CV_64FC3);
            Scalar tpgradb = Scalar(0.0, 0.0, 0.0);
            vector<string> convKey = getKeys(nsamples, cl, j, KEY_UP_DELTA);
            for(int m = 0; m < convKey.size(); m ++){
                Mat temp = rot90(cpmap.at(convKey[m]), 2);
                if(cl == 0){
                    tpgradW += convCalc(x[getSampleNum(convKey[m])], temp, CONV_VALID);
                }else{
                    string strprev = getPreviousLayerKey(convKey[m], KEY_POOL);
                    tpgradW += convCalc(cpmap.at(strprev), temp, CONV_VALID);
                }
                tpgradb += sum(cpmap.at(convKey[m]));
            }
            if(convConfig[cl].is3chKernel){
                CLayers[cl].layer[j].Wgrad = tpgradW / nsamples + convConfig[cl].WeightDecay * CLayers[cl].layer[j].W;
                CLayers[cl].layer[j].bgrad = tpgradb.mul(1 / nsamples);
            }else{
                vector<Mat> _tpgradWs;
                split(tpgradW, _tpgradWs);
                Mat _bufferW = Mat::zeros(convConfig[cl].KernelSize, convConfig[cl].KernelSize, CV_64FC1);
                double _bufferb = 0.0;
                for(int ch = 0; ch < _tpgradWs.size(); ch++){
                    _bufferW += _tpgradWs[ch];
                    _bufferb += tpgradb[ch];
                }
                _bufferW = _bufferW / nsamples + convConfig[cl].WeightDecay * _tpgradWs.size() * CLayers[cl].layer[j].W;
                _bufferb /= nsamples;
                CLayers[cl].layer[j].Wgrad = _bufferW;
                CLayers[cl].layer[j].bgrad = Scalar(_bufferb);
                _tpgradWs.clear();
                _bufferW.release();
            }
            tpgradW.release();
        }
    }
    // destructor
    p.release();
    M.release();
    groundTruth.release();
    cpmap.clear();
    locmap.clear();
    factor.clear();
    acti.clear();
    delta.clear();
    bernoulli.clear();
}








