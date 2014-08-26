#include "helper.h"

using namespace cv;
using namespace std;


// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s = ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

Vec3d Scalar2Vec3d(Scalar a){
    Vec3d res(a[0], a[1], a[2]);
    return res;
}

Scalar Vec3d2Scalar(Vec3d a){
	return Scalar(a[0], a[1], a[2]);
} 


void
unconcatenateMat(vector<Mat> &src, vector<vector<Mat> > &dst, int vsize){
    for(int i = 0; i < src.size() / vsize; i++){
        vector<Mat> tmp;
        for(int j = 0; j < vsize; j++){
            Mat img = src[i * vsize + j];
            vector<Mat> imgs;
            split(img, imgs);
            for(int ch = 0; ch < imgs.size(); ch++){
                tmp.push_back(imgs[ch]);
            }
        }
        dst.push_back(tmp);
    }
}

Mat 
concatenateMat(vector<vector<Mat> > &vec){
    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    int height = vec[0].size() * subFeatures;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[i].size(); j++){
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat 
concatenateMat(vector<Mat> &vec, int matcols){
    vector<vector<Mat> > temp;
    unconcatenateMat(vec, temp, vec.size() / matcols);
    return concatenateMat(temp);
}

double 
getLearningRate(Mat &data){
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;
    int nsamples = data.cols;
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;
    SVD uwvT = SVD(Sigma);
    Sigma.release();
    return 0.9 / uwvT.w.ATD(0, 0);
}

void 
getSample(vector<Mat>& src, vector<Mat>& dst, int _size){
    dst.clear();
    if(src.size() < _size){
        for(int i = 0; i < src.size(); i++){
            dst.push_back(src[i]);
        }
        return;
    }
    int randomNum = ((long)rand() + (long)rand()) % (src.size() - _size - 1);
    for(int i = 0; i < _size; i++){
        dst.push_back(src[i + randomNum]);
    }
}

void 
getSample(Mat& src, Mat& dst, int _size){
    if(src.cols < _size){
        Rect roi = Rect(0, 0, src.cols, src.rows);
        src(roi).copyTo(dst);
        return;
    }
    int randomNum = ((long)rand() + (long)rand()) % (src.cols - _size - 1);
    Rect roi = Rect(randomNum, 0, _size, src.rows);
    src(roi).copyTo(dst);
}

void 
getSample(vector<Mat>& src1, vector<Mat>& dst1, Mat& src2, Mat& dst2, int _size){
    dst1.clear();
    if(src1.size() < _size){
        for(int i = 0; i < src1.size(); i++){
            dst1.push_back(src1[i]);
        }
        Rect roi = Rect(0, 0, src2.cols, src2.rows);
        src2(roi).copyTo(dst2);
        return;
    }
    int randomNum = ((long)rand() + (long)rand()) % (src2.cols - _size - 1);
    for(int i = 0; i < _size; i++){
        dst1.push_back(src1[i + randomNum]);
    }
    Rect roi = Rect(randomNum, 0, _size, src2.rows);
    src2(roi).copyTo(dst2);
}





