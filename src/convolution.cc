#include "convolution.h"


Point
findLoc(const Mat &prob, int m){
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
findLocCh3(const Mat &prob, int m){
    vector<Mat> probs;
    split(prob , probs);
    vector<Point> res;
    for(int i = 0; i < probs.size(); i++){
        res.push_back(findLoc(probs[i], m));
    }
    return res;
}

void
minMaxLoc(const Mat &img, Scalar &minVal, Scalar &maxVal, vector<Point> &minLoc, vector<Point> &maxLoc){
    vector<Mat> imgs;
    split(img, imgs);
    for(int i = 0; i < imgs.size(); i++){
        Point min;
        Point max;
        double minval;
        double maxval;
        minMaxLoc( imgs[i], &minval, &maxval, &min, &max);
        minLoc.push_back(min);
        maxLoc.push_back(max);
        minVal[i] = minval;
        maxVal[i] = maxval;
    }
}

Mat
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod, vector<vector<Point> > &locat){
    if(pVert == 1 && pHori == 1){
        vector<Point> tppt;
        for(int ch = 0; ch < 3; ch++){
            tppt.push_back(Point(0, 0));
        }
        locat.push_back(tppt);
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
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
                for(int ch = 0; ch < 3; ch++){
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
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod){
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
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
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp).mul(1 / (pVert * pHori));
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                Scalar sumval = sum(temp);
                Mat prob = temp.mul(Reciprocal(sumval));
                val = sum(prob.mul(temp));
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
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    Mat res;
    if(POOL_MEAN == poolingMethod){
        Mat one = cv::Mat(pVert, pHori, CV_64FC3, Scalar(1.0, 1.0, 1.0));
        res = kron(M, one) / (pVert * pHori);
        one.release();
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC3);
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                for(int ch = 0; ch < 3; ch++){
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
        from = (current_kernel_num - lrn_size / 2) >= 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) <= (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale / (to - from + 1) + Scalar::all(1.0);
    divide(res, pow(sum, lrn_beta), res);
    sum.release();
    return res;
}

Mat
localResponseNorm(const vector<vector<Mat> > &vec, int cl, int k, int s, int m){

    vector<Mat>::const_iterator first = vec[s].begin() + m * convConfig[cl].KernelAmount;
    vector<Mat>::const_iterator last = vec[s].begin() + (m + 1) * convConfig[cl].KernelAmount;
    vector<Mat> current_layer(first, last);

    Mat res;
    vec[s][m * convConfig[cl].KernelAmount + k].copyTo(res);
    Mat sum = Mat::zeros(res.rows, res.cols, CV_64FC3);
    int from, to;
    if(convConfig[cl].KernelAmount < lrn_size){
        from = 0;
        to = convConfig[cl].KernelAmount - 1;
    }else{
        from = (k - lrn_size / 2) >= 0 ? (k - lrn_size / 2) : 0;
        to = (k + lrn_size / 2) <= (convConfig[cl].KernelAmount - 1) ? 
             (k + lrn_size / 2) : (convConfig[cl].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        Mat tmpmat;
        current_layer[j].copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale / (to - from + 1) + Scalar::all(1.0);
    divide(res, pow(sum, lrn_beta), res);
    sum.release();
    current_layer.clear();
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
        from = (current_kernel_num - lrn_size / 2) >= 0 ? (current_kernel_num - lrn_size / 2) : 0;
        to = (current_kernel_num + lrn_size / 2) <= (convConfig[current_layer_num].KernelAmount - 1) ? 
             (current_kernel_num + lrn_size / 2) : (convConfig[current_layer_num].KernelAmount - 1);
    }
    for(int j = from; j <= to; j++){
        string tmpstr = current_layer + "K" + i2str(j);
        Mat tmpmat;
        map.at(tmpstr).copyTo(tmpmat);
        sum += pow(tmpmat, 2.0);
        tmpmat.release();
    }
    sum = sum * lrn_scale / (to - from + 1) + Scalar::all(1.0);

    res = da.mul(pow(sum, lrn_beta));
    tmp = a.mul(a).mul(da);
    tmp = tmp.mul(pow(sum, lrn_beta - 1)) * lrn_scale / (to - from + 1) * 2.0;
    res -= tmp;

    pow(sum, lrn_beta * 2, tmp);
    res = divide(res, pow(sum, lrn_beta * 2));
    a.release();
    da.release();
    sum.release();
    tmp.release();
    return res;
}

void 
convAndPooling(const vector<Mat> &x, const vector<Cvl> &CLayers, 
                unordered_map<string, Mat> &map, 
                unordered_map<string, vector<vector<Point> > > &loc){
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
                    else tmpconv += Scalar(CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0]);
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
                    tmpconv = Pooling(tmpconv, pdim, pdim, pooling_method, PoolingLoc);
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
                        else tmpconv += Scalar(CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0]);
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
                        tmpconv = Pooling(tmpconv, pdim, pdim, pooling_method, PoolingLoc);
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
hashDelta(const Mat &src, unordered_map<string, Mat> &map, int layersize, int type){
    int nsamples = src.cols;
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vecstr;
        for(int i = 0; i < layersize; i ++){
            if(i == 0){
                string s2 = s1 + "C0";
                for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                    string s3 = s2 + "K" + i2str(k) + "P";
                    if(i == layersize - 1){
                        if(type == HASH_DELTA) s3 += "D";
                        elif(type == HASH_HESSIAN) s3 += "H";
                    }
                    vecstr.push_back(s3);
                }
            }else{
                vector<string> vec2;
                for(int j = 0; j < vecstr.size(); j ++){
                    string s2 = vecstr[j] + "C" + i2str(i);
                    for(int k = 0; k < convConfig[i].KernelAmount; k ++){
                        string s3 = s2 + "K" + i2str(k) + "P";
                        if(i == layersize - 1){
                            if(type == HASH_DELTA) s3 += "D";
                            elif(type == HASH_HESSIAN) s3 += "H";
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

void 
convAndPooling(const vector<Mat> &x, const vector<Cvl> &CLayers, vector<vector<Mat> > &res){

    int nsamples = x.size();
    res.clear();
    for(int i = 0; i < nsamples; i++){
        vector<Mat> tmp;
        tmp.push_back(x[i]);
        res.push_back(tmp);
    }
    vector<vector<Mat> > tpvec(nsamples);
    for(int cl = 0; cl < convConfig.size(); cl++){
        for(int i = 0; i < tpvec.size(); i++){
            tpvec[i].clear();
        }
        int pdim = convConfig[cl].PoolingDim;
        for(int s = 0; s < nsamples; s++){
            for(int m = 0; m < res[s].size(); m++){
                for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = convCalc(res[s][m], temp, CONV_VALID);
                    if(convConfig[cl].is3chKernel) tmpconv += CLayers[cl].layer[k].b;
                    else tmpconv += Scalar(CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0], CLayers[cl].layer[k].b[0]);
                    tmpconv = nonLinearityC3(tmpconv);
                    tpvec[s].push_back(tmpconv);
                }
                for(int k = 0; k < convConfig[cl].KernelAmount; k++){
                    Mat temp = tpvec[s][m * convConfig[cl].KernelAmount + k];
                    if(convConfig[cl].useLRN) temp = localResponseNorm(tpvec, cl, k, s, m);
                    tpvec[s][m * convConfig[cl].KernelAmount + k] = Pooling(temp, pdim, pdim, pooling_method);
                }
            }
        }
        swap(res, tpvec);
    }   
    for(int i = 0; i < tpvec.size(); i++){
        tpvec[i].clear();
    }  
    tpvec.clear();
}
