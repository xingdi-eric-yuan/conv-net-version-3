#conv-net-version-3.0
=====================

Newly implemented convnet (c++).

This is an **EARLY ADOPTERS EDITION**, which is still buggy, I'll post formal version in days.

To run this code, you should have 
* a cifar-10 dataset;
* OpenCV.

##File description (todo)

##Compile & Run

* Compile: "cmake CMakeLists.txt" and then "make" 
 
* Run: "./conv" 

##Updates 

* 3-channels images supported.
* Add Dropout;
* In conv layers, one can use either 3-channel conv kernels or single-chanel conv kernels (that is to say, whether share weights).
* Local Response Normalization supported (still buggy).
