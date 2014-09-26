#include "read_config.h"

using namespace std;

string 
read_2_string(string File_name){
    char *pBuf;
    FILE *pFile = NULL;   
    if(!(pFile = fopen(File_name.c_str(),"r"))){
        printf("Can not find this file.");
        return 0;
    }
    //move pointer to the end of the file
    fseek(pFile, 0, SEEK_END);
    //Gets the current position of a file pointer.offset 
    int len = ftell(pFile);
    pBuf = new char[len];
    //Repositions the file pointer to the beginning of a file
    rewind(pFile);
    fread(pBuf, 1, len, pFile);
    fclose(pFile);
    string res = pBuf;
    return res;
}

bool
get_word_bool(string &str, string name){

    size_t pos = str.find(name);    
    int i = pos + 1;
    bool res = true;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        if(!content.compare("true")) res = true;
        else res = false;
    }
    str.erase(pos, i - pos + 1);
    return res;
}

int
get_word_int(string &str, string name){

    size_t pos = str.find(name);    
    int i = pos + 1;
    int res = 1;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = stoi(content);
    }
    str.erase(pos, i - pos + 1);
    return res;
}

double
get_word_double(string &str, string name){

    size_t pos = str.find(name);    
    int i = pos + 1;
    double res = 0.0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = atof(content.c_str());
    }
    str.erase(pos, i - pos + 1);
    return res;
}

int
get_word_type(string &str, string name){

    size_t pos = str.find(name);    
    int i = pos + 1;
    int res = 0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        if(!content.compare("POOL_MAX")) res = 0;
        elif(!content.compare("NL_SIGMOID")) res = 0;
        elif(!content.compare("CONV")) res = 0;
        elif(!content.compare("POOL_MEAN")) res = 1;
        elif(!content.compare("NL_TANH")) res = 1;
        elif(!content.compare("FC")) res = 1;
        elif(!content.compare("POOL_STOCHASTIC")) res = 2;
        elif(!content.compare("NL_RELU")) res = 2;
        elif(!content.compare("SOFTMAX")) res = 2;
    }
    str.erase(pos, i - pos + 1);
    return res;
}

void 
delete_comment(string &str){
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1){
        if(head == str.length()) break;
        if(str[head] == '/'){
            tail = head + 1;
            while(1){
                if(tail == str.length()) break;
                if(str[tail] == '/') break;
                ++ tail;
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
}

void 
delete_space(string &str){
    if(str.empty()) return;
    int i = 0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == '\t' || str[i] == '\n' || str[i] == ' '){
            str.erase(str.begin() + i);
        }else ++ i;
    }
}

void 
get_layers_config(string &str){
    vector<string> layers;
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1){
        if(head == str.length()) break;
        if(str[head] == '$'){
            tail = head + 1;
            while(1){
                if(tail == str.length()) break;
                if(str[tail] == '&') break;
                ++ tail;
            }
            string sub = str.substr(head, tail - head + 1);
            if(sub[sub.length() - 1] == '&'){
                sub.erase(sub.begin() + sub.length() - 1);
                sub.erase(sub.begin());
                layers.push_back(sub);
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
    for(int i = 0; i < layers.size(); i++){
        int type = get_word_type(layers[i], "LAYER");
        switch(type){
            case 0:{
                int ks = get_word_int(layers[i], "KERNEL_SIZE");
                int ka = get_word_int(layers[i], "KERNEL_AMOUNT");
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");
                int pd = get_word_int(layers[i], "POOLING_DIM");
                bool is3ch = get_word_bool(layers[i], "IS_3CH_KERNEL");
                bool lrn = get_word_bool(layers[i], "USE_LRN");
                convConfig.push_back(ConvLayerConfig(ks, ka, wd, pd, is3ch, lrn));
                break;
            }case 1:{
                int hn = get_word_int(layers[i], "NUM_HIDDEN_NEURONS");
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");
                double dor = get_word_double(layers[i], "DROPOUT_RATE");
                fcConfig.push_back(FullConnectLayerConfig(hn, wd, dor));
                break;
            }case 2:{
                softmaxConfig.NumClasses = get_word_int(layers[i], "NUM_CLASSES");
                softmaxConfig.WeightDecay = get_word_double(layers[i], "WEIGHT_DECAY");
                break;
            }
        }
    }
}

void
readConfigFile(string filepath){

    string str = read_2_string(filepath);
    delete_space(str);
    delete_comment(str);
    get_layers_config(str);

    is_gradient_checking = get_word_bool(str, "IS_GRADIENT_CHECKING");
    if(is_gradient_checking){
        for(int i = 0; i < fcConfig.size(); i++){
            fcConfig[i].DropoutRate = 1.0;
        }
    }
    use_log = get_word_bool(str, "USE_LOG");
    batch_size = get_word_int(str, "BATCH_SIZE");
    pooling_method = get_word_type(str, "POOLING_METHOD");
    non_linearity = get_word_type(str, "NON_LINEARITY");

    training_epochs = get_word_int(str, "TRAINING_EPOCHS");
    lrate_w = get_word_double(str, "LRATE_W");
    lrate_b = get_word_double(str, "LRATE_B");
    iter_per_epo = get_word_int(str, "ITER_PER_EPO");
    lrate_decay = get_word_double(str, "LRATE_DECAY");

    cout<<"****************************************************************************"<<endl
        <<"**                    READ CONFIG FILE COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;

    for(int i = 0; i < convConfig.size(); i++){
        cout<<"***** convolutional layer: "<<i<<" *****"<<endl;
        cout<<"KernelSize = "<<convConfig[i].KernelSize<<endl;
        cout<<"KernelAmount = "<<convConfig[i].KernelAmount<<endl;
        cout<<"WeightDecay = "<<convConfig[i].WeightDecay<<endl;
        cout<<"PoolingDim = "<<convConfig[i].PoolingDim<<endl;
        cout<<"is3chKernel = "<<convConfig[i].is3chKernel<<endl;
        cout<<"useLRN = "<<convConfig[i].useLRN<<endl<<endl;
    }
    for(int i = 0; i < fcConfig.size(); i++){

        cout<<"***** full-connected layer: "<<i<<" *****"<<endl;
        cout<<"NumHiddenNeurons = "<<fcConfig[i].NumHiddenNeurons<<endl;
        cout<<"WeightDecay = "<<fcConfig[i].WeightDecay<<endl;
        cout<<"DropoutRate = "<<fcConfig[i].DropoutRate<<endl<<endl;
    }
    cout<<"***** softmax layer: *****"<<endl;
    cout<<"NumClasses = "<<softmaxConfig.NumClasses<<endl;
    cout<<"WeightDecay = "<<softmaxConfig.WeightDecay<<endl<<endl;
    cout<<"***** general config *****"<<endl;
    cout<<"is_gradient_checking = "<<is_gradient_checking<<endl;
    cout<<"use_log = "<<use_log<<endl;
    cout<<"batch size = "<<batch_size<<endl;
    cout<<"pooling method = "<<pooling_method<<endl;
    cout<<"non-linearity method = "<<non_linearity<<endl;
    cout<<"training epochs = "<<training_epochs<<endl;
    cout<<"learning rate for weight matrices = "<<lrate_w<<endl;
    cout<<"learning rate for bias = "<<lrate_b<<endl;
    cout<<"iteration per epoch = "<<iter_per_epo<<endl;
    cout<<"learning rate decay = "<<lrate_decay<<endl<<endl;
}
