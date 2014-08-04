#conv-net-version-3.0
=====================

Newly implemented convnet (C++ / OpenCV).

This is an **EARLY ADOPTERS EDITION**, which is still buggy, I'll post formal version in days.

To run this code, you should have 
* a cifar-10 dataset;
* OpenCV.

##Compile & Run

* Compile: "cmake CMakeLists.txt" and then "make" 
 
* Run: "./conv" 

##Updates 

* 3-channels images supported.
* Add Dropout;
* In conv layers, one can use either 3-channel conv kernels or single-chanel conv kernels (that is to say, whether share weights).
* Local Response Normalization supported.

##File description 

* **channel_3.h & channel_3.cc** - functions dealing with 3-channels matrices

* **convolution.h & convolution.cc** - convolution / pooling / local response normalization implementations

* **cost_gradient.h & cost_gradient.cc** - function which calculates the cost function and gradients given data and weights of each layer

* **gradient_checking.h & gradient_checking.cc** - gradient checking functions. (You may want to disable dropout during gradient checking.)

* **helper.h & helper.cc** - tiny helper functions

* **matrix_maths.h & matrix_maths.cc** - matrix maths functions, such as conv2() and rot90()

* **read_data.h & read_data.cc** - this read_data supports only CIFAR-10 dataset

* **result_predict.h & result_predict.cc** - functions for result predict

* **save_weights.h & save_weights.cc** - save weights into .txt file

* **string_proc.h & string_proc.cc** - I'm using unordered_map<string, Mat>, so here are the tedious string processing functions

* **train_network.h & train_network.cc** - using sgd with momentum method

* **weight_init.h & weight_init.cc** - initialize weights of the whole network

* **general_settings.h** - general settings

* **data_structure.h** - data structure definition of network

* **sample.cc** - main() inside :)
