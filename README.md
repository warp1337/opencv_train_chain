### OPENCV TRAIN CHAIN

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

Detect any kind of objects using 2d object detection and GPU acceleration
using OpenCV and CUDA

![OPENCV TRAIN CHAIN](https://github.com/CentralLabFacilities/clf_2d_gpu_detect/blob/master/clf_gpu_detect_screenshot.png "")

### Installation

Install the latest nvidia cuda toolkit: https://developer.nvidia.com/cuda-toolkit
This version (master) has been tested with version 7.5

    HowTo: http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu

Install OpenCV minimum 3.1.0 with CUDA support

    HowTo: http://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.htm

Invoke OpenCV's cmake with:

    -DWITH_CUDA=ON

Then:

    git clone https://github.com/warp1337/opencv_train_chain.git
    cd opencv_train_chain
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX={your decision} ..
    make
    make install

### Usage

Collect images

    otc-collect-data --help


An exemplary config file can be found in the cfg folder.

    ./otc-opencv-orb /path/to/configfile/example.yaml

### License

See README.md
