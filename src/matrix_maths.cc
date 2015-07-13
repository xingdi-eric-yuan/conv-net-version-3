#include "matrix_maths.h"

using namespace cv;
using namespace std;

Scalar 
Reciprocal(const Scalar &s){
    Scalar res;
    divide(Scalar(1.0, 1.0, 1.0), s, res);
    return res;
}

Mat 
Reciprocal(const Mat &M){
    if(M.channels() == 1) return 1.0 / M;
    else{
        Mat one = Mat(M.size(), CV_64FC3, Scalar(1.0, 1.0, 1.0));
        return divide(one, M);
    }
}

Mat 
sigmoid(const Mat &M){
    Mat tmp = exp(-M) + 1.0;
    return div(1.0, tmp);
}

Mat 
dsigmoid_a(const Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat 
dsigmoid(const Mat &M){
    Mat tmp = exp(M);
    Mat tmp2 = tmp + 1.0;
    tmp2 = pow(tmp2, 2.0);
    return divide(tmp, tmp2);
}

Mat
ReLU(const Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
    res = res.mul(M);
    return res;
}

Mat
dReLU(const Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
    return res;
}

Mat
LeakyReLU(const Mat& M){
    Mat p = M > 0.0;
    p.convertTo(p, CV_64FC1, 1.0 / 255.0, 0);
    p = p.mul(M);
    Mat n = M < 0.0;
    n.convertTo(n, CV_64FC1, 1.0 / 255.0, 0);
    n = n.mul(M);
    n = n.mul(1 / leaky_relu_alpha);
    return p + n;
}

Mat
dLeakyReLU(const Mat& M){
    Mat p = M > 0.0;
    p.convertTo(p, CV_64FC1, 1.0 / 255.0, 0);
    Mat n = M < 0.0;
    n.convertTo(n, CV_64FC1, 1.0 / 255.0, 0);
    n = n.mul(1 / leaky_relu_alpha);
    return p + n;
}

Mat 
Tanh(const Mat &M){
    Mat res;
    M.copyTo(res);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            res.ATD(i, j) = tanh(M.ATD(i, j));
        }
    }
    return res;
}

Mat
dTanh(const Mat &M){
    Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
    return res - M.mul(M);
}

Mat 
nonLinearity(const Mat &M, int method){
    if(method == NL_RELU){
        return ReLU(M);
    }elif(method == NL_TANH){
        return Tanh(M);
    }elif(method == NL_LEAKY_RELU){
        return LeakyReLU(M);
    }else{
        return sigmoid(M);
    }
}

Mat 
dnonLinearity(const Mat &M, int method){
    if(method == NL_RELU){
        return dReLU(M);
    }elif(method == NL_TANH){
        return dTanh(M);
    }elif(method == NL_LEAKY_RELU){
        return dLeakyReLU(M);
    }else{
        return dsigmoid(M);
    }
}

// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(const Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}

Mat 
conv2(const Mat &img, const Mat &kernel, int convtype, int padding, int stride) {
    Mat tmp;
    Mat source = img;
    // padding
    source = Mat();
    copyMakeBorder(img, source, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));
    
    // zero padding for CONV_FULL
    int additionalRows, additionalCols;
    if(CONV_FULL == convtype) {
        additionalRows = kernel.rows - 1;
        additionalCols = kernel.cols - 1;
        copyMakeBorder(source, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
    } 
    Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
    int borderMode = BORDER_CONSTANT;
    Mat fkernel;
    flip(kernel, fkernel, -1);
    filter2D(source, tmp, img.depth(), fkernel, anchor, 0, borderMode);
    // cut matrix for CONV_VALID
    if(CONV_VALID == convtype) {
        tmp = tmp.colRange((kernel.cols - 1) / 2, tmp.cols - kernel.cols / 2)
                   .rowRange((kernel.rows - 1) / 2, tmp.rows - kernel.rows / 2);
    }
    int xsize = tmp.cols / stride;
    if(tmp.cols % stride > 0) ++xsize;
    int ysize = tmp.rows / stride;
    if(tmp.rows % stride > 0) ++ysize;
    Mat dest = Mat::zeros(ysize, xsize, CV_64FC1);
    for(int i = 0; i < dest.rows; i++){
        for(int j = 0; j < dest.cols; j++){
            dest.ATD(i, j) = tmp.ATD(i * stride, j * stride);
        }
    }
    return dest;
}

Mat convolveDFT(const Mat &A, const Mat &B){
    Mat C;
    // reallocate the output array if needed
    C.create(abs(A.rows + B.rows)-1, abs(A.cols + B.cols)-1, A.type());
    Size dftSize;
    // calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
    // allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));
    Mat tempB(dftSize, B.type(), Scalar::all(0));
    // copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
    A.copyTo(roiA);
    Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
    B.copyTo(roiB);
    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    dft(tempA, tempA, 0, A.rows);
    dft(tempB, tempB, 0, B.rows);
    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA, 0, false);
    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
    //idft(tempA, tempA, DFT_SCALE, A.rows + B.rows - 1);
    // now copy the result back to C.
    tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
    return C;
    // all the temporary buffers will be deallocated automatically
}

UMat convolveDFT(const UMat &A, const UMat &B){
    UMat C;
    // reallocate the output array if needed
    C.create(abs(A.rows + B.rows)-1, abs(A.cols + B.cols)-1, A.type());
    Size dftSize;
    // calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
    // allocate temporary buffers and initialize them with 0's
    UMat tempA(dftSize, A.type(), Scalar::all(0));
    UMat tempB(dftSize, B.type(), Scalar::all(0));
    // copy A and B to the top-left corners of tempA and tempB, respectively
    UMat roiA(tempA, Rect(0,0,A.cols,A.rows));
    A.copyTo(roiA);
    UMat roiB(tempB, Rect(0,0,B.cols,B.rows));
    B.copyTo(roiB);
    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    dft(tempA, tempA, 0, A.rows);
    dft(tempB, tempB, 0, B.rows);
    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA, 0, false);
    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
    //idft(tempA, tempA, DFT_SCALE, A.rows + B.rows - 1);
    // now copy the result back to C.
    tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
    return C;
    // all the temporary buffers will be deallocated automatically
}

Mat 
conv2dft(const Mat &img, const Mat &kernel, int convtype, int padding, int stride) {
    Mat tmp;
    Mat source;

    img.copyTo(tmp);
    // padding
    source = Mat();
    copyMakeBorder(tmp, source, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));
    /*
    UMat usource, ukernel, uconv;
    source.copyTo(usource);
    kernel.copyTo(ukernel);
    uconv = convolveDFT(usource, ukernel);
    uconv.copyTo(tmp);
    */
    tmp = convolveDFT(source, kernel);


    if(CONV_SAME == convtype){
        tmp = tmp.colRange((kernel.cols) / 2, tmp.cols - kernel.cols / 2)
                   .rowRange((kernel.rows) / 2, tmp.rows - kernel.rows / 2);
    }
    if(CONV_VALID == convtype) {
        int tmpx = source.cols - kernel.cols + 1;
        int tmpy = source.rows - kernel.rows + 1;
        tmp = tmp.colRange((tmp.cols - tmpx) / 2, tmp.cols - ((tmp.cols - tmpx) / 2))
                   .rowRange((tmp.rows - tmpy) / 2, tmp.rows - ((tmp.cols - tmpx) / 2));
    }
    int xsize = tmp.cols / stride;
    if(tmp.cols % stride > 0) ++xsize;
    int ysize = tmp.rows / stride;
    if(tmp.rows % stride > 0) ++ysize;
    Mat dest = Mat::zeros(ysize, xsize, CV_64FC1);
    for(int i = 0; i < dest.rows; i++){
        for(int j = 0; j < dest.cols; j++){
            dest.ATD(i, j) = tmp.ATD(i * stride, j * stride);
        }
    }
    return dest;
}


Mat
convCalc(const Mat &img, const Mat &kernel, int convtype, int padding, int stride){
    Mat tmp;
    img.copyTo(tmp);
    if(tmp.channels() == 1 && kernel.channels() == 1){
        return conv2dft(tmp, kernel, convtype, padding, stride);
    }else{
        return parallel3(conv2dft, tmp, kernel, convtype, padding, stride);
    }
}

Mat 
doPadding(Mat &src, int pad){
    Mat res;
    copyMakeBorder(src, res, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0));
    return res;
}

Mat 
dePadding(Mat &src, int pad){
    Mat res;
    src(Rect(pad, pad, src.cols - pad * 2, src.rows - pad * 2)).copyTo(res);
    return res;
}

Mat 
interpolation(Mat &src, Mat &sizemat){

    int stride = sizemat.rows / src.rows;
    if(sizemat.rows % src.rows > 0) ++ stride;
    if(stride == 0 || stride == 1) return src;
    Mat res = Mat::zeros(sizemat.size(), CV_64FC3);
    for(int i = 0; i < src.rows; i ++){
        for(int j = 0; j < src.cols; j ++){
            res.AT3D(i * stride, j * stride) = src.AT3D(i, j);
        }
    }
    return res;
}

Mat 
interpolation(Mat &src, int _size){
    int stride = _size / src.rows;
    if(_size % src.rows > 0) ++ stride;
    //cout<<src.rows<<", "<<_size<<", "<<stride<<endl;
    if(stride == 0 || stride == 1) return src;
    Mat res = Mat::zeros(_size, _size, CV_64FC3);
    for(int i = 0; i < src.rows; i ++){
        for(int j = 0; j < src.cols; j ++){
            res.AT3D(i * stride, j * stride) = src.AT3D(i, j);
        }
    }
    return res;
}

// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat
kron(const Mat &a, const Mat &b){
    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC3);
    Mat c;
    vector<Mat> bs;
    vector<Mat> cs;
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            bs.clear();
            cs.clear();
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            split(b , bs);
            for(int ch = 0; ch < 3; ch++){
                cs.push_back(bs[ch].mul(a.AT3D(i, j)[ch]));
            }
            merge(cs, c);
            c.copyTo(temp);
        }
    }
    bs.clear();
    vector<Mat>().swap(bs);
    cs.clear();
    vector<Mat>().swap(cs);
    return res;
}

Mat 
getBernoulliMatrix(int height, int width, double prob){
    // randu builds a Uniformly distributed matrix
    Mat ran = Mat::zeros(height, width, CV_64FC1);
    randu(ran, Scalar(0), Scalar(1.0));
    Mat res = ran >= prob;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    return res;
}

// Follows are OpenCV maths
Mat 
exp(const Mat &src){
    Mat dst;
    exp(src, dst);
    return dst;
}

Mat 
div(double x, const Mat &src){
    Mat dst;
    src.copyTo(dst);
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(src.channels() == 3){
               for(int ch = 0; ch < 3; ch++){
                    if(dst.AT3D(i, j)[ch] != 0.0) dst.AT3D(i, j)[ch] = x / dst.AT3D(i, j)[ch];
                }
            }else{
                if(dst.ATD(i, j) != 0.0) dst.ATD(i, j) = x / dst.ATD(i, j);
            }
        }
    }
    return dst;
}

Mat 
div(const Mat &src, double x){
    if(x == 0.0) return src;
    Mat dst;
    src.copyTo(dst);
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(src.channels() == 3){
               for(int ch = 0; ch < 3; ch++){
                    dst.AT3D(i, j)[ch] = dst.AT3D(i, j)[ch] / x;
                }
            }else{
                dst.ATD(i, j) = dst.ATD(i, j) / x;
            }
        }
    }
    return dst;
}

Scalar 
div(const Scalar &src, double x){
    if(x == 0.0) return src;
    Scalar dst(0.0, 0.0, 0.0);
    for(int ch = 0; ch < 3; ch++){
        dst[ch] = src[ch] / x;
    }
    return dst;
}

Mat 
log(const Mat &src){
    Mat dst;
    log(src, dst);
    return dst;
}

Mat 
reduce(const Mat &src, int direc, int conf){
    Mat dst;
    reduce(src, dst, direc, conf);
    return dst;
}

Mat 
divide(const Mat &m1, const Mat &m2){
    Mat dst;
    divide(m1, m2, dst);
    return dst;
}

Mat 
pow(const Mat &m1, double val){
    Mat dst;
    pow(m1, val, dst);
    return dst;
}

double 
sum1(const Mat &m){
    double res = 0.0;
    Scalar tmp = sum(m);
    for(int i = 0; i < m.channels(); i++){
        res += tmp[i];
    }
    return res;
}

double
max(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return maxval;
}

double
min(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return minval;
}

// Pooling with overlap
// Max pooling and stochastic pooling supported
// output size = (input size - window size) / stride + 1
Mat 
Pooling_with_overlap(const Mat &M, Size2i window_size, int stride, int poolingMethod, std::vector<std::vector<Point> > &locat){
    Mat tmpres = Mat::zeros(M.rows - window_size.height + 1, M.cols - window_size.width + 1, CV_64FC3);
    std::vector<std::vector<Point> > tmplocat;
    for(int i = 0; i < M.rows - window_size.height + 1; ++i){
        for(int j = 0; j < M.cols - window_size.width + 1; ++j){
            Mat tmp;
            M(Rect(j, i, window_size.width, window_size.height)).copyTo(tmp);

            Scalar val = Scalar(0.0, 0.0, 0.0);
            std::vector<Point> tppt;
            if(POOL_MAX == poolingMethod){
                Scalar minVal = Scalar(0.0, 0.0, 0.0);
                Scalar maxVal = Scalar(0.0, 0.0, 0.0);
                std::vector<Point> minLoc; 
                std::vector<Point> maxLoc;
                minMaxLoc(tmp, minVal, maxVal, minLoc, maxLoc );
                val = maxVal;
                for(int ch = 0; ch < 3; ch++){
                    tppt.push_back(Point(maxLoc[ch].x + j, maxLoc[ch].y + i));
                }
            }elif(POOL_STOCHASTIC == poolingMethod){
                
                Scalar recip_sumval = sum(tmp);
                divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
                Mat prob = tmp.mul(recip_sumval);
                int ran = rand() % (tmp.rows * tmp.cols);
                std::vector<Point> loc = findLocCh3(prob, ran);
                for(int ch = 0; ch < loc.size(); ch++){
                    val[ch] = tmp.AT3D(loc[ch].y, loc[ch].x)[ch];
                    tppt.push_back(Point(loc[ch].x + j, loc[ch].y + i));
                }     
            }
            tmplocat.push_back(tppt);  
            tmpres.AT3D(i, j) = Scalar2Vec3d(val);
            tppt.clear();
            std::vector<Point>().swap(tppt);
        }
    }
    int xsize = tmpres.cols / stride;
    if(tmpres.cols % stride > 0) ++xsize;
    int ysize = tmpres.rows / stride;
    if(tmpres.rows % stride > 0) ++ysize;
    Mat dest = Mat::zeros(ysize, xsize, CV_64FC3);

    for(int i = 0; i < tmpres.rows; i++){
        for(int j = 0; j < tmpres.cols; j++){
            if(i % stride > 0 || j % stride > 0) continue;
            for(int ch = 0; ch < 3; ++ch){
                dest.AT3D(i / stride, j / stride)[ch] = tmpres.AT3D(i, j)[ch];
            }
            locat.push_back(tmplocat[i * tmpres.cols + j]);
        }
    }
    tmplocat.clear();
    std::vector<std::vector<Point> >().swap(tmplocat);
    return dest;
}

Mat 
Pooling_with_overlap_test(const Mat &M, Size2i window_size, int stride, int poolingMethod){
    Mat tmpres = Mat::zeros(M.rows - window_size.height + 1, M.cols - window_size.width + 1, CV_64FC3);
    std::vector<std::vector<Point> > tmplocat;
    for(int i = 0; i < M.rows - window_size.height + 1; ++i){
        for(int j = 0; j < M.cols - window_size.width + 1; ++j){
            Mat tmp;
            M(Rect(j, i, window_size.width, window_size.height)).copyTo(tmp);
            Scalar val = Scalar(0.0, 0.0, 0.0);
            if(POOL_MAX == poolingMethod){
                Scalar minVal = Scalar(0.0, 0.0, 0.0);
                Scalar maxVal = Scalar(0.0, 0.0, 0.0);
                std::vector<Point> minLoc; 
                std::vector<Point> maxLoc;
                minMaxLoc(tmp, minVal, maxVal, minLoc, maxLoc );
                val = maxVal;
            }elif(POOL_STOCHASTIC == poolingMethod){
                
                Scalar recip_sumval = sum(tmp);
                divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
                Mat prob = tmp.mul(recip_sumval);
                int ran = rand() % (tmp.rows * tmp.cols);
                std::vector<Point> loc = findLocCh3(prob, ran);
                for(int ch = 0; ch < loc.size(); ch++){
                    val[ch] = tmp.AT3D(loc[ch].y, loc[ch].x)[ch];
                }     
            }
            tmpres.AT3D(i, j) = Scalar2Vec3d(val);
        }
    }
    int xsize = tmpres.cols / stride;
    if(tmpres.cols % stride > 0) ++xsize;
    int ysize = tmpres.rows / stride;
    if(tmpres.rows % stride > 0) ++ysize;
    Mat dest = Mat::zeros(ysize, xsize, CV_64FC3);

    for(int i = 0; i < tmpres.rows; i++){
        for(int j = 0; j < tmpres.cols; j++){
            if(i % stride > 0 || j % stride > 0) continue;
            for(int ch = 0; ch < 3; ++ch){
                dest.AT3D(i / stride, j / stride)[ch] = tmpres.AT3D(i, j)[ch];
            }
        }
    }
    return dest;
}


// Max pooling and stochastic pooling supported
Mat 
UnPooling_with_overlap(const Mat &M, Size2i window_size, int stride, int poolingMethod, std::vector<std::vector<Point> > &locat, Size2i up_size){
    Mat res;
    if(window_size.height == 1 && window_size.width == 1 && stride == 1){
        M.copyTo(res);
        return res;
    }
    res = Mat::zeros(up_size, CV_64FC3);
    for(int i = 0; i < M.rows; i++){
        for(int j = 0; j < M.cols; j++){
            for(int ch = 0; ch < 3; ch++){
                res.AT3D(locat[i * M.cols + j][ch].y, locat[i * M.cols + j][ch].x)[ch] += M.AT3D(i, j)[ch];
            }
        }
    }
    return res;
}

Mat
Pooling(const Mat &M, int stride, int poolingMethod, std::vector<std::vector<Point> > &locat){
    if(stride == 1){
        std::vector<Point> tppt;
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                tppt.clear();
                for(int ch = 0; ch < 3; ch++){
                    tppt.push_back(Point(j, i));
                }
                locat.push_back(tppt);
            }
        }
        Mat res;
        M.copyTo(res);
        return res;
    }
    Mat newM;
    M.copyTo(newM);
    Mat res = Mat::zeros(newM.rows / stride, newM.cols / stride, CV_64FC3);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * stride, i * stride, stride, stride);
            newM(roi).copyTo(temp);
            Scalar val = Scalar(0.0, 0.0, 0.0);
            std::vector<Point> tppt;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                Scalar minVal = Scalar(0.0, 0.0, 0.0);
                Scalar maxVal = Scalar(0.0, 0.0, 0.0);
                std::vector<Point> minLoc; 
                std::vector<Point> maxLoc;
                minMaxLoc( temp, minVal, maxVal, minLoc, maxLoc );
                val = maxVal;
                for(int ch = 0; ch < 3; ch++){
                    tppt.push_back(Point(maxLoc[ch].x + j * stride, maxLoc[ch].y + i * stride));
                }
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                double recip = 1.0 / (stride * stride);
                val = sum(temp).mul(Scalar(recip, recip, recip));
                for(int ch = 0; ch < 3; ch++){
                    tppt.push_back(Point(j * stride, i * stride));
                }
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                Scalar recip_sumval = sum(temp);
                divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
                Mat prob = temp.mul(recip_sumval);
                int ran = rand() % (temp.rows * temp.cols);
                std::vector<Point> loc = findLocCh3(prob, ran);
                for(int ch = 0; ch < loc.size(); ch++){
                    val[ch] = temp.AT3D(loc[ch].y, loc[ch].x)[ch];
                    tppt.push_back(Point(loc[ch].x + j * stride, loc[ch].y + i * stride));
                }
            }
            res.AT3D(i, j) = Scalar2Vec3d(val);
            locat.push_back(tppt);
        }
    }
    return res;
}

Mat
Pooling_test(const Mat &M, int stride, int poolingMethod){
    if(stride == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    Mat newM;
    M.copyTo(newM);
    Mat res = Mat::zeros(newM.rows / stride, newM.cols / stride, CV_64FC3);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * stride, i * stride, stride, stride);
            newM(roi).copyTo(temp);
            Scalar val = Scalar(0.0, 0.0, 0.0);
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                Scalar minVal = Scalar(0.0, 0.0, 0.0);
                Scalar maxVal = Scalar(0.0, 0.0, 0.0);
                std::vector<Point> minLoc; 
                std::vector<Point> maxLoc;
                minMaxLoc( temp, minVal, maxVal, minLoc, maxLoc );
                val = maxVal;
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                double recip = 1.0 / (stride * stride);
                val = sum(temp).mul(Scalar(recip, recip, recip));
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                Scalar recip_sumval = sum(temp);
                divide(Scalar(1.0, 1.0, 1.0), recip_sumval, recip_sumval);
                Mat prob = temp.mul(recip_sumval);
                int ran = rand() % (temp.rows * temp.cols);
                std::vector<Point> loc = findLocCh3(prob, ran);
                for(int ch = 0; ch < loc.size(); ch++){
                    val[ch] = temp.AT3D(loc[ch].y, loc[ch].x)[ch];
                }
            }
            res.AT3D(i, j) = Scalar2Vec3d(val);
        }
    }
    return res;
}

Mat 
UnPooling(const Mat &M, int stride, int poolingMethod, std::vector<std::vector<Point> > &locat, Size2i up_size){
    Mat res;
    if(stride == 1){
        M.copyTo(res);
        return res;
    }
    if(POOL_MEAN == poolingMethod){

        Mat one = cv::Mat(stride, stride, CV_64FC3, Scalar(1.0, 1.0, 1.0));
        cout<<M.size()<<",     "<<one.size()<<endl;
        res = kron(M, one);
        divide(res, Scalar(stride * stride, stride * stride, stride * stride), res);
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * stride, M.cols * stride, CV_64FC3);
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                for(int ch = 0; ch < 3; ch++){
                    res.AT3D(locat[i * M.cols + j][ch].y, locat[i * M.cols + j][ch].x)[ch] = M.AT3D(i, j)[ch];
                }
            }
        }
    }
    copyMakeBorder(res, res, 0, up_size.height - res.rows, 0, up_size.width - res.cols, BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0));
    return res;
}

Point 
findLoc(const Mat &prob, int m){
    Mat temp, idx;
    Point res = Point(0, 0);
    prob.reshape(0, 1).copyTo(temp); 
    sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
    int i = idx.at<int>(0, m);
    res.x = i % prob.cols;
    res.y = i / prob.cols;
    return res;
}

std::vector<Point> 
findLocCh3(const Mat &prob, int m){
    std::vector<Mat> probs;
    split(prob , probs);
    std::vector<Point> res;
    for(int i = 0; i < probs.size(); i++){
        res.push_back(findLoc(probs[i], m));
    }
    probs.clear();
    std::vector<Mat>().swap(probs);
    return res;
}

Mat 
findMax(const Mat &M){
    Mat tmp;
    M.copyTo(tmp);
    Mat result = Mat::zeros(1, tmp.cols, CV_64FC1);
    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < tmp.cols; i++){
        minMaxLoc(tmp(Rect(i, 0, 1, tmp.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (double) maxLoc.y;
    }
    return result;
}


void 
minMaxLoc(const Mat &img, Scalar &minVal, Scalar &maxVal, std::vector<Point> &minLoc, std::vector<Point> &maxLoc){
    std::vector<Mat> imgs;
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

int 
compareMatrix(const Mat &a, const Mat &b){
    Mat tmp;
    b.copyTo(tmp);
    tmp -= a;
    Mat res = (tmp == 0.0);
    res.convertTo(res, CV_64FC1, 1.0 / 255.0, 0);
    return (int)(sum1(res));
}



