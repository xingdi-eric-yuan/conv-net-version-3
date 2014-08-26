#include "get_sample.h"

using namespace cv;
using namespace std;

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
getSample(Mat& src, Mat& dst, int _size, int _type){
    if(_type == SAMPLE_ROWS){
        if(src.rows < _size){
            Rect roi = Rect(0, 0, src.cols, src.rows);
            src(roi).copyTo(dst);
            return;
        }
        int randomNum = ((long)rand() + (long)rand()) % (src.rows - _size - 1);
        Rect roi = Rect(0, randomNum, src.cols, _size);
        src(roi).copyTo(dst);
    }else{
        if(src.cols < _size){
            Rect roi = Rect(0, 0, src.cols, src.rows);
            src(roi).copyTo(dst);
            return;
        }
        int randomNum = ((long)rand() + (long)rand()) % (src.cols - _size - 1);
        Rect roi = Rect(randomNum, 0, _size, src.rows);
        src(roi).copyTo(dst);
    }
}

void 
getSample(vector<Mat>& src1, vector<Mat>& dst1, Mat& src2, Mat& dst2, int _size, int _type){
    dst1.clear();
    if(_type == SAMPLE_ROWS){
        if(src1.size() < _size){
            for(int i = 0; i < src1.size(); i++){
                dst1.push_back(src1[i]);
            }
            Rect roi = Rect(0, 0, src2.cols, src2.rows);
            src2(roi).copyTo(dst2);
            return;
        }
        int randomNum = ((long)rand() + (long)rand()) % (src2.rows - _size - 1);
        for(int i = 0; i < _size; i++){
            dst1.push_back(src1[i + randomNum]);
        }
        Rect roi = Rect(0, randomNum, src2.cols, _size);
        src2(roi).copyTo(dst2);
    }else{
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
}





