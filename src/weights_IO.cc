#include "weights_IO.h"
using namespace cv;
using namespace std;

void 
save2XML(string path, string name, std::vector<network_layer*>& flow){

    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string tmp = path + "/" + name + ".xml";
    FileStorage fs(tmp, FileStorage::WRITE);
    int conv = 0;
    int fc = 0;

    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "convolutional"){
            tmp = "convolutional_layer_" + std::to_string(conv);
            for(int j = 0; j < ((convolutional_layer*)flow[i]) -> kernels.size(); j++){
                string tmp2 = tmp + "_kernel_" + std::to_string(j);
                fs << (tmp2 + "_w") << ((convolutional_layer*)flow[i]) -> kernels[j] -> w;
            }
            ++ conv;
        }elif(flow[i] -> layer_type == "fully_connected"){
            tmp = "fully_connected_layer_" + std::to_string(fc);
            fs << (tmp + "_w") << ((fully_connected_layer*)flow[i]) -> w;
            ++ fc;
        }elif(flow[i] -> layer_type == "softmax"){
            fs << ("softmax_layer_w") << ((softmax_layer*)flow[i]) -> w;
        }
    }    
    fs.release();
    cout<<"Successfully saved network information..."<<endl;
}
