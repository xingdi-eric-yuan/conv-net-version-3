#include "converter.h"

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

Scalar makeScalar(double m){
    Scalar res = Scalar(m, m, m);
    return res;
}

// convert from vector of img to matrix
// vec.size() == nsamples
void convert(std::vector<std::vector<Mat> >& vec, Mat &M){
    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    Mat res = Mat::zeros(3 * vec[0].size() * subFeatures, vec.size(), CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        for(int m = 0; m < vec[i].size(); m++){

            std::vector<Mat> tmpvec;
            split(vec[i][m], tmpvec);
            for(int j = 0; j < tmpvec.size(); j++){
                Rect roi = Rect(i, m * 3 * subFeatures + j * subFeatures, 1, subFeatures);
                Mat subView = res(roi);
                Mat tmp = tmpvec[j].reshape(0, subFeatures);
                tmp.copyTo(subView);
            }
        }
    }
    res.copyTo(M);
}

// convert from matrix to vector of img
// vec.size() == nsamples
void convert(Mat &M, std::vector<std::vector<Mat> >& vec, int nsamples, int imagesize){
    std::vector<Mat> tmpvec;
    for(int i = 0; i < nsamples; i++){
        tmpvec.clear();
        int dim = imagesize * imagesize;
        vector<Mat> mats;
        for(int j = 0; j < M.rows; j += dim * 3){
            mats.clear();
            for(int k = 0; k < 3; k++){
                Mat tmp;
                M(Rect(i, j + k * dim, 1, dim)).copyTo(tmp);
                tmp = tmp.reshape(0, imagesize);
                mats.push_back(tmp);
            }
            Mat res;
            merge(mats, res);
            tmpvec.push_back(res);
        }
        vec.push_back(tmpvec);
    }
    tmpvec.clear();
    std::vector<Mat>().swap(tmpvec);
}

