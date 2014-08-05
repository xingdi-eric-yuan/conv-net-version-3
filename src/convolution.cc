#include "convolution.h"


Point
findLoc(Mat &prob, int m){
    Mat temp, idx;
    Point res = Point(0, 0);
    prob.reshape(0, 1).copyTo(temp); 
    sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
    int i = idx.at<int>(0, m);
    res.x = i % prob.rows;
    res.y = i / prob.rows;
    temp.release();
    idx.release();
    return res;
}

vector<Point>
findLocCh3(Mat &prob, int m){
    vector<Mat> probs;
    split(prob , probs);
    vector<Point> res;
    for(int i = 0; i < probs.size(); i++){
        res.push_back(findLoc(probs[i], m));
    }
    return res;
}

void
minMaxLoc(Mat &img, Scalar &minVal, Scalar &maxVal, vector<Point> &minLoc, vector<Point> &maxLoc){
    vector<Mat> imgs;
    split(img, imgs);
    for(int i = 0; i < imgs.size(); i++){
        Point min;
        Point max;
        minMaxLoc( imgs[i], &minVal[i], &maxVal[i], &min, &max);
        minLoc.push_back(min);
        maxLoc.push_back(max);
    }
}

Mat
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod, vector<vector<Point> > &locat, bool isTest){
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX, remY, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC3);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            Scalar val = Scalar(0.0, 0.0, 0.0);
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                Scalar minVal = Scalar(0.0, 0.0, 0.0);
                Scalar maxVal = Scalar(0.0, 0.0, 0.0);
                vector<Point> minLoc; 
                vector<Point> maxLoc;
                minMaxLoc( temp, minVal, maxVal, minLoc, maxLoc );
                val = maxVal;
                vector<Point> tppt;
                for(int ch = 0; ch < minLoc.size(); ch++){
                    tppt.push_back(Point(maxLoc[ch].x + j * pHori, maxLoc[ch].y + i * pVert));
                }
                locat.push_back(tppt);
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp).mul(1 / (pVert * pHori));
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                Scalar sumval = sum(temp);
                Mat prob = temp.mul(Reciprocal(sumval));
                if(isTest){
                    val = sum(prob.mul(temp));
                }else{
                    int ran = rand() % (temp.rows * temp.cols);
                    vector<Point> loc = findLocCh3(prob, ran);
                    for(int ch = 0; ch < loc.size(); ch++){
                        val[ch] = temp.AT3D(loc[ch].y, loc[ch].x)[ch];
                    }
                    vector<Point> tppt;
                    for(int ch = 0; ch < loc.size(); ch++){
                        tppt.push_back(Point(loc[ch].x + j * pHori, loc[ch].y + i * pVert));
                    }
                    locat.push_back(tppt);
                }
                prob.release();
            }
            res.AT3D(i, j) = Scalar2Vec3d(val);
            temp.release();
        }
    }
    newM.release();
    return res;
}

Mat 
UnPooling(const Mat &M, int pVert, int pHori, int poolingMethod, vector<vector<Point> > &locat){
    Mat res;
    if(POOL_MEAN == poolingMethod){
        Mat one = Mat::ones(pVert, pHori, CV_64FC3);
        res = kron(M, one) / (pVert * pHori);
        one.release();
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC3);
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                for(int ch = 0; ch < locat[0].size(); ch++){
                    res.AT3D(locat[i * M.cols + j][ch].y, locat[i * M.cols + j][ch].x)[ch] = M.AT3D(i, j)[ch];
                }
            }
        }
    }
    return res;
}

Mat
localResponseNorm(const unordered_map<string, Mat> &map, string str){

    int current_kernel_num = getCurrentKernelNum(str);
    int current_layer_num = getCurrentLayerNum(str);
    string current_layer = getCurrentLayer(str);

    Mat res;
    map.at(current_layer + "K" + i2str(current_kernel_num)).copyTo(res);
    Mat sum = Mat::zeros(res.rows, res.cols, CV_64FC3);

    int from, to;
    if(convConfig[current_layer_num].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[current_layer_num].KernelAmount - 1;
    }else{
        from = (current_kernel_num - lrn_size / 2) > 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) < (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        pow(tmpmat, 2.0, tmpmat);
        sum += tmpmat;
        tmpmat.release();
    }
    sum = sum * lrn_scale / convConfig[current_layer_num].KernelAmount;
    sum += Scalar(1.0, 1.0, 1.0);
    pow(sum, lrn_beta, sum);
    divide(res, sum, res);
    sum.release();
    return res;
}

Mat
dlocalResponseNorm(const unordered_map<string, Mat> &map, string str){

    int current_kernel_num = getCurrentKernelNum(str);
    int current_layer_num = getCurrentLayerNum(str);
    string current_layer = getCurrentLayer(str);

    Mat a, da, tmp, res;
    map.at(current_layer + "K" + i2str(current_kernel_num)).copyTo(a);
    map.at(str).copyTo(da);
    Mat sum = Mat::zeros(a.rows, a.cols, CV_64FC3);
    int from, to;
    if(convConfig[current_layer_num].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[current_layer_num].KernelAmount - 1;
    }else{
        from = (current_kernel_num - lrn_size / 2) > 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) < (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        pow(tmpmat, 2, tmpmat);
        sum += tmpmat;
        tmpmat.release();
    }
    sum = sum * lrn_scale / convConfig[current_layer_num].KernelAmount;
    sum += Scalar(1.0, 1.0, 1.0);

    pow(sum, lrn_beta, tmp);
    res = da.mul(tmp);

    pow(sum, lrn_beta - 1, tmp);
    tmp = tmp.mul(a);
    tmp = tmp.mul(a);
    tmp = tmp.mul(da);
    tmp *= lrn_scale / convConfig[current_layer_num].KernelAmount * 2.0;
    res -= tmp;

    pow(sum, lrn_beta * 2, tmp);
    divide(res, tmp, res);
    a.release();
    da.release();
    sum.release();
    tmp.release();
    return res;
}



void 
convAndPooling(const vector<Mat> &x, const vector<Cvl> &CLayers, 
                unordered_map<string, Mat> &map, 
                unordered_map<string, vector<vector<Point> > > &loc, bool isTest){
    // Conv & Pooling
    int nsamples = x.size();
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vec;
        for(int cl = 0; cl < CLayers.size(); cl ++){
            int pdim = convConfig[cl].PoolingDim;
            if(cl == 0){
                // Convolution
                for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = convCalc(x[m], temp, CONV_VALID);
                    if(convConfig[cl].is3chKernel) tmpconv += CLayers[cl].layer[k].b;
                    else tmpconv += CLayers[cl].layer[k].b[0];
                    tmpconv = nonLinearityC3(tmpconv);
                    map[s2] = tmpconv;
                    temp.release();
                    tmpconv.release();
                }
                // Local response normalization & Pooling
                for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    // Local response normalization
                    Mat tmpconv;
                    map.at(s2).copyTo(tmpconv);
                    if(convConfig[cl].useLRN) tmpconv = localResponseNorm(map, s2);
                    vector<vector<Point> > PoolingLoc;
                    tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, isTest);
                    string s3 = s2 + "P";
                    map[s3] = tmpconv;
                    loc[s3] = PoolingLoc;
                    vec.push_back(s3);
                    tmpconv.release();
                    PoolingLoc.clear();
                }
            }else{
                vector<string> tmpvec;
                for(int tp = 0; tp < vec.size(); tp ++){
                    // Convolution
                    for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                        Mat tmpconv = convCalc(map.at(vec[tp]), temp, CONV_VALID);
                        if(convConfig[cl].is3chKernel) tmpconv += CLayers[cl].layer[k].b;
                        else tmpconv += CLayers[cl].layer[k].b[0];
                        tmpconv = nonLinearityC3(tmpconv);
                        map[s2] = tmpconv;
                        temp.release();
                        tmpconv.release();
                    }
                    // Local response normalization & Pooling
                    for(int k = 0; k < convConfig[cl].KernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat tmpconv;
                        map.at(s2).copyTo(tmpconv);
                        if(convConfig[cl].useLRN) tmpconv = localResponseNorm(map, s2);
                        vector<vector<Point> > PoolingLoc;
                        tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, isTest);
                        string s3 = s2 + "P";
                        map[s3] = tmpconv;
                        loc[s3] = PoolingLoc;
                        tmpvec.push_back(s3);
                        tmpconv.release();
                        PoolingLoc.clear();
                    }
                    
                }
                swap(vec, tmpvec);
                tmpvec.clear();
            }
        }    
        vec.clear();   
    }
}

void
hashDelta(const Mat &src, unordered_map<string, Mat> &map, vector<Cvl> &CLayers){
    int nsamples = src.cols;
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vecstr;
        for(int i = 0; i < CLayers.size(); i ++){
            if(i == 0){
                string s2 = s1 + "C0";
                for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                    string s3 = s2 + "K" + i2str(k) + "P";
                    if(i == CLayers.size() - 1){
                        s3 += "D";
                    }
                    vecstr.push_back(s3);
                }
            }else{
                vector<string> vec2;
                for(int j = 0; j < vecstr.size(); j ++){
                    string s2 = vecstr[j] + "C" + i2str(i);
                    for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                        string s3 = s2 + "K" + i2str(k) + "P";
                        if(i == CLayers.size() - 1){
                            s3 += "D";
                        }
                        vec2.push_back(s3);
                    }
                }
                swap(vecstr, vec2);
                vec2.clear();
            }
        }
        int sqDim = src.rows / vecstr.size() / 3;
        int Dim = sqrt(sqDim);
        for(int i = 0; i < vecstr.size(); i++){
            Mat color = Mat::zeros(Dim, Dim, CV_64FC3);
            vector<Mat> channels;
            for(int ch = 0; ch < 3; ch++){
                Rect roi = Rect(m, 3 * i * sqDim + sqDim * ch, 1, sqDim);
                Mat temp;
                src(roi).copyTo(temp);
                Mat img = temp.reshape(0, Dim);
                channels.push_back(img);
            }
            merge(channels, color);
            map[vecstr[i]] = color;
            channels.clear();
        }  
    }
}





