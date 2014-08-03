#include "string_proc.h"

//#define KEY_CONV 0
//#define KEY_POOL 1
//#define KEY_DELTA 2
//#define KEY_UP_DELTA 3
using namespace std;

void
toKey(string &str, int keyType){
    if(keyType == KEY_CONV) ;//do nothing
    elif(keyType == KEY_POOL) str += "P";
    elif(keyType == KEY_DELTA) str += "PD";
    elif(keyType == KEY_UP_DELTA) str += "PUD";
}

vector<string>
getKeys(int nsamples, int end_layer, int spec_kernel, int keyType){
    vector<string> vecstr;
    for(int i = 0; i < nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j = 0; j <= end_layer; j++){
        vector<string> tmpvecstr;
        for(int i = 0; i < vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            if(j == end_layer){
                int k = spec_kernel;
                string s3 = s2 + "K" + i2str(k);
                toKey(s3, keyType);
                tmpvecstr.push_back(s3);
            }else{
                for(int k = 0; k < convConfig[j].KernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    s3 += "P";
                    tmpvecstr.push_back(s3);
                }
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

vector<string>
getSpecKeys(int nsamples, int end_layer, int spec_layer, int spec_kernel, int keyType){
    if(end_layer == spec_layer) return getKeys(nsamples, end_layer, spec_kernel, keyType);
    vector<string> vecstr;
    for(int i=0; i<nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j=0; j<=end_layer; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            if(j == end_layer){
                for(int k = 0; k < convConfig[j].KernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    toKey(s3, keyType);
                    tmpvecstr.push_back(s3);
                }
            }elif(j == spec_layer){
                int k = spec_kernel;
                string s3 = s2 + "K" + i2str(k) + "P";
                tmpvecstr.push_back(s3);
            }else{
                for(int k = 0; k < convConfig[j].KernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    s3 += "P";
                    tmpvecstr.push_back(s3);
                }
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

vector<string>
getLayerKey(int nsamples, int layer, int keyType){
    vector<string> vecstr;
    for(int i = 0; i < nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j = 0; j <= layer; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            for(int k = 0; k < convConfig[j].KernelAmount; k++){
                string s3 = s2 + "K" + i2str(k);
                if(j != layer){
                    s3 += "P";
                }else{
                    if(keyType == KEY_POOL){
                        s3 += "P";
                    }elif(keyType == KEY_DELTA){
                        s3 += "PD";
                    }elif(keyType == KEY_UP_DELTA){
                        s3 += "PUD";
                    }
                }
                tmpvecstr.push_back(s3);
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

vector<string>
getLayer(int nsamples, int layer){
    vector<string> vecstr;
    for(int i = 0; i < nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j = 0; j < layer; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            for(int k = 0; k < convConfig[j].KernelAmount; k++){
                string s3 = s2 + "K" + i2str(k) + "P";
                tmpvecstr.push_back(s3);
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    for(int i = 0; i < vecstr.size(); i++){
        vecstr[i] += "C" + i2str(layer);
    }
    return vecstr;
}

int 
getSampleNum(string str){
    int i = 1;
    while(str[i] >='0' && str[i] <= '9'){
        ++ i;
    }
    string sub = str.substr(1, i - 1);
    return str2i(sub);
}

int 
getCurrentKernelNum(string str){
    int i = str.length() - 1;
    while(str[i] !='K'){
        -- i;
    }
    int start = i + 1;
    i = start;
    while(str[i] <= '9' && str[i] >= '0'){
        ++ i;
    }
    string sub = str.substr(start, i - start);
    return str2i(sub);
}

int
getCurrentLayerNum(string str){
    int i = str.length() - 1; 
    while(str[i] != 'K'){
        -- i;
    }
    int j = i;
    while(str[j] != 'C'){
        -- j;
    }
    string res = str.substr(j + 1, i - j - 1);
    return str2i(res);
}

string
getCurrentLayer(string str){
    int i = str.length() - 1; 
    while(str[i] != 'K'){
        -- i;
    }
    string res = str.substr(0, i);
    return res;
}

string
getPreviousLayerKey(string str, int keyType){
    int i = str.length() - 1; 
    while(str[i] != 'C'){
        -- i;
    }
    string res = str.substr(0, i - 1);
    toKey(res, keyType);
    return res;
}









