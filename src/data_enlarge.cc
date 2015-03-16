#include "data_enlarge.h"

using namespace cv;
using namespace std;

void
fliplr(const Mat &_from, Mat &_to){
    flip(_from, _to, 1);
}

void
flipud(const Mat &_from, Mat &_to){
    flip(_from, _to, 0);
}

void
flipudlr(const Mat &_from, Mat &_to){
    flip(_from, _to, -1);
}

void
rotateNScale(const Mat &_from, Mat &_to, double angle, double scale){
    Point center = Point(_from.cols / 2, _from.rows / 2);
   // Get the rotation matrix with the specifications above
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
   // Rotate the warped image
    warpAffine(_from, _to, rot_mat, _to.size());
}

void
addWhiteNoise(const Mat &_from, Mat &_to, double stdev){

    _to = Mat::zeros(_from.rows, _from.cols, CV_64FC3);
    randu(_to, Scalar(-1.0, -1.0, -1.0), Scalar(1.0, 1.0, 1.0));
    _to *= stdev;
    _to += _from;
    // how to make this faster?
    for(int i = 0; i < _to.rows; i++){
        for(int j = 0; j < _to.cols; j++){
			Vec3d rgb = _to.at<Vec3d>(i, j);
			for(int c = 0; c < _to.channels(); c++){
				if(rgb.val[c] < 0.0) rgb.val[c] = 0.0;
				if(rgb.val[c] > 1.0) rgb.val[c] = 1.0;
			}
			_to.at<Vec3d>(i, j) = rgb;
        }
    }
}

void 
dataEnlarge(vector<Mat>& data, Mat& label){
    int nSamples = data.size();
    
    /*
    // flip left right
    for(int i = 0; i < nSamples; i++){
        fliplr(data[i], tmp);
        data.push_back(tmp);
    }
    // flip up down
    for(int i = 0; i < nSamples; i++){
        flipud(data[i], tmp);
        data.push_back(tmp);
    }
    // flip left right up down
    for(int i = 0; i < nSamples; i++){
        flipudlr(data[i], tmp);
        data.push_back(tmp);
    }
    */
    // rotate -10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, -10, 1.2);
        data.push_back(tmp);
    }
    // rotate +10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, 10, 1.2);
        data.push_back(tmp);
    }
    // add white noise
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        addWhiteNoise(data[i], tmp, 0.1);
        data.push_back(tmp);
    }
    // copy label matrix
	cv::Mat tmp;
    repeat(label, 1, 4, tmp); 
	label = tmp;
}





