#include "read_data.h"

using namespace cv;
using namespace std;

void 
read_batch(string filename, vector<Mat> &vec, Mat &label){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels(3);
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                tp.copyTo(channels[ch]);
//                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.ATD(0, i) = (double)tplabel;
        }
    }
	file.close();
}

void
read_CIFAR10_data(vector<Mat> &trainX, vector<Mat> &testX, Mat &trainY, Mat &testY){
    string filename;
    filename = "cifar-10-batches-bin/data_batch_";
    vector<Mat> labels;
    vector<vector<Mat> > batches;
	int number_batch = 5;
    for(int i = 1; i <= number_batch; i++){
        vector<Mat> tpbatch;
        Mat tplabel = Mat::zeros(1, 10000, CV_64FC1);   
        string name = filename + std::to_string((long long)i) + ".bin";
        read_batch(name, tpbatch, tplabel);
        labels.push_back(tplabel);
        batches.push_back(tpbatch);
        tpbatch.clear();
    }
    // trainX
    trainX.reserve(batches[0].size() * number_batch);
    for(int i = 0; i < number_batch; i++){
        trainX.insert(trainX.end(), batches[i].begin(), batches[i].end());
    }
    // trainY
    trainY = Mat::zeros(1, 10000 * number_batch, CV_64FC1);
    Rect roi;
    Mat subView;
    for(int i = 0; i < number_batch; i++){
        roi = cv::Rect(labels[i].cols * i, 0, labels[i].cols, 1);
        subView = trainY(roi);
        labels[i].copyTo(subView);
    }
    subView.release();
    // testX, testY
    filename = "cifar-10-batches-bin/test_batch.bin";
    testY = Mat::zeros(1, 10000, CV_64FC1);  
	
    read_batch(filename, testX, testY);
    preProcessing(trainX, testX);
    dataEnlarge(trainX, trainY);

    cout<<"****************************************************************************"<<endl
        <<"**                        READ DATASET COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;
    cout<<"The training data has "<<trainX.size()<<" images, each images has "<<trainX[0].cols<<" columns and "<<trainX[0].rows<<" rows."<<endl;
    cout<<"The testing data has "<<testX.size()<<" images, each images has "<<testX[0].cols<<" columns and "<<testX[0].rows<<" rows."<<endl;
    cout<<"There are "<<trainY.cols<<" training labels and "<<testY.cols<<" testing labels."<<endl<<endl;
}


Mat 
concat(const vector<Mat> &vec){
    int height = vec[0].rows * vec[0].cols;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC3);
    for(int i = 0; i < vec.size(); i++){
        Rect roi = Rect(i, 0, 1, height);
        Mat subView = res(roi);
        Mat ptmat = vec[i].reshape(0, height);
        ptmat.copyTo(subView);
    }
    return res;
}
//*/

void
preProcessing(vector<Mat> &trainX, vector<Mat> &testX){
    for(int i = 0; i < trainX.size(); i++){
        //cvtColor(trainX[i], trainX[i], CV_RGB2YCrCb);
        trainX[i].convertTo(trainX[i], CV_64FC3, 1.0/255, 0);
    }
    for(int i = 0; i < testX.size(); i++){
        //cvtColor(testX[i], testX[i], CV_RGB2YCrCb);
        testX[i].convertTo(testX[i], CV_64FC3, 1.0/255, 0);
    }
    
    /*
    for(int i = 0; i < trainX.size(); i++)
        trainX[i] -= 0.5;
    for(int i = 0; i < testX.size(); i++)
        testX[i] -= 0.5;
*/

    // first convert vec of mat into a single mat
    Mat tmp = concat(trainX);
    Mat tmp2 = concat(testX);
    Mat alldata = Mat::zeros(tmp.rows, tmp.cols + tmp2.cols, CV_64FC3);
    
    tmp.copyTo(alldata(Rect(0, 0, tmp.cols, tmp.rows)));
    tmp2.copyTo(alldata(Rect(tmp.cols, 0, tmp2.cols, tmp.rows)));

    Scalar mean;
    Scalar stddev;
    meanStdDev(alldata, mean, stddev);

    for(int i = 0; i < trainX.size(); i++){
        divide(trainX[i] - mean, stddev, trainX[i]);
    }
    for(int i = 0; i < testX.size(); i++){
        divide(testX[i] - mean, stddev, testX[i]);
    }
    tmp.release();
    tmp2.release();
    alldata.release();


/*
    // get average
    Mat average = trainX[0];
    for(int i = 1; i < trainX.size(); i++) average += trainX[i];
    for(int i = 0; i < testX.size(); i++) average += testX[i];
    
    average = div(average, (trainX.size() + testX.size()));
    // subtract average
    for(int i = 0; i < trainX.size(); i++)
        trainX[i] -= average;
    for(int i = 0; i < testX.size(); i++)
        testX[i] -= average;
*/
}





