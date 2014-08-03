#include "matrix_maths.h"

using namespace cv;
using namespace std;

Scalar 
Reciprocal(Scalar &s){
    Scalar res = Scalar(1.0, 1.0, 1.0);
    for(int i = 0; i < 3; i++){
        res[i] = res[i] / s[i];
    }
    return res;
}

Mat 
Reciprocal(Mat &M){
    return 1.0 / M;
}

Mat 
sigmoid(Mat &M){
    Mat temp;
    exp(-M, temp);
    return 1.0 / (temp + 1.0);
}

Mat 
dsigmoid(Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat
ReLU(Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    res = res.mul(M);
    return res;
}

Mat
dReLU(Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255.0);
    return res;
}

Mat 
Tanh(Mat &M){
    Mat res(M);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            res.ATD(i, j) = tanh(M.ATD(i, j));
        }
    }
    return res;
}

Mat
dTanh(Mat &M){
    Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
    return res - M.mul(M);
}

Mat 
nonLinearity(Mat &M){
    if(nonlin == NL_RELU){
        return ReLU(M);
    }elif(nonlin == NL_TANH){
        return Tanh(M);
    }else{
        return sigmoid(M);
    }
}

Mat 
dnonLinearity(Mat &M){
    if(nonlin == NL_RELU){
        return dReLU(M);
    }elif(nonlin == NL_TANH){
        return dTanh(M);
    }else{
        return dsigmoid(M);
    }
}

Mat 
nonLinearityC3(Mat &M){
	return parallel3(nonLinearity, M);
}

Mat 
dnonLinearityC3(Mat &M){
	return parallel3(dnonLinearity, M);
}


// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}

// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat 
conv2(Mat &img, Mat &kernel, int convtype) {
    Mat dest;
    Mat source = img;
    if(CONV_FULL == convtype) {
        source = Mat();
        int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
        copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
    }
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    int borderMode = BORDER_CONSTANT;
    Mat fkernal;
    flip(kernel, fkernal, -1);
    filter2D(source, dest, img.depth(), fkernal, anchor, 0, borderMode);

    if(CONV_VALID == convtype) {
        dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
                   .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
    }
    source.release();
    fkernal.release();
    return dest;
}

Mat
convCalc(Mat &img, Mat &kernel, int convtype){
    if(img.channels() == 1 && kernel.channels() == 1){
        return conv2(img, kernel, convtype);
    }else{
        return parallel3(conv2, img, kernel, convtype);
    }
}

// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat
kron(Mat &a, Mat &b){
    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC3);
    for(int i=0; i<a.rows; i++){
        for(int j=0; j<a.cols; j++){
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.AT3D(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}

Mat 
getBernoulliMatrix(int height, int width, double prob){
    // randu builds a Uniformly distributed matrix
    Mat ran = Mat::zeros(height, width, CV_64FC1);
    randu(ran, Scalar(0), Scalar(1.0));
    Mat res = ran <= prob;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    ran.release();
    return res;
}
















