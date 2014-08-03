#include "save_weights.h"

using namespace cv;
using namespace std;


void
save2txt3ch(Mat &data, string str, int step){
    string s = std::to_string(step);
    str += s;
    string str_r = str + "_r";
    string str_g = str + "_g";
    string str_b = str + "_b";
    str_r += ".txt";
    str_g += ".txt";
    str_b += ".txt";
    FILE *pOut_r = fopen(str_r.c_str(), "w");
    FILE *pOut_g = fopen(str_g.c_str(), "w");
    FILE *pOut_b = fopen(str_b.c_str(), "w");
    for(int i=0; i<data.rows; i++){
        for(int j=0; j<data.cols; j++){
            Vec3d tmp = data.AT3D(i, j);
            fprintf(pOut_r, "%lf", tmp[0]);
            fprintf(pOut_g, "%lf", tmp[1]);
            fprintf(pOut_b, "%lf", tmp[2]);
            if(j == data.cols - 1){
                fprintf(pOut_r, "\n");
                fprintf(pOut_g, "\n");
                fprintf(pOut_b, "\n");
            } 
            else{
                fprintf(pOut_r, " ");
                fprintf(pOut_g, " ");
                fprintf(pOut_b, " ");
            } 
        }
    }
    fclose(pOut_r);
    fclose(pOut_g);
    fclose(pOut_b);
}

void
save2txt(Mat &data, string str, int step){
    string s = std::to_string(step);
    str += s;
    string str_r = str + "_r";
    str_r += ".txt";
    FILE *pOut_r = fopen(str_r.c_str(), "w");
    for(int i=0; i<data.rows; i++){
        for(int j=0; j<data.cols; j++){
            double tmp = data.ATD(i, j);
            fprintf(pOut_r, "%lf", tmp);
            if(j == data.cols - 1){
                fprintf(pOut_r, "\n");
            } 
            else{
                fprintf(pOut_r, " ");
            } 
        }
    }
    fclose(pOut_r);
}

void
mkdir(vector<Cvl> &CLayers){
    int layers = CLayers.size();
    for(int i = 0; i < layers; i++){
        string str = "./weight/cl_" + std::to_string(i);
        mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        int kernels = convConfig[i].KernelAmount;
        for(int j = 0; j < kernels; j++){
            string str2 = str + "/kernel_" + std::to_string(j);
            mkdir(str2.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
    }
}

void
save2txt(vector<Cvl> &CLayers, int step){
    for(int cl = 0; cl < CLayers.size(); cl++){
        for(int i=0; i<convConfig[cl].KernelAmount; i++){
            string str = "weight/cl_" + std::to_string(cl) + "/kernel_" + std::to_string(i) + "/epoch_";
            if(convConfig[cl].is3chKernel) save2txt3ch(CLayers[cl].layer[i].W, str, step);
            else save2txt(CLayers[cl].layer[i].W, str, step);
        }
    }
}

