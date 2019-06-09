# Installing Deep Learning Frameworks on Ubuntu with CUDA support

![Tensorflow CUDA](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/tensorflow_with_cuda.png)

![Pytorch Keras](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/keras_pytorch.jpeg)

In this article, we will learn how to install Deep Learning Frameworks like TensorFlow, Keras and PyTorch on a machine having a NVIDIA graphics card.

We will install **CUDA**, **cuDNN**, **Python 2**, **Python 3**, **TensorFlow**, **Keras**, **Pytorch**, **OpenCV**, **Dlib** along with other Python Machine Learning libraries step-by-step.

We have tested the instructions on a system with the following configuration:

![System Info](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/system_info.png)

### Step 1 : Install Prerequisites

Before installing anything, let us first update the information about the packages stored on the computer and upgrade the already installed packages to their latest versions.

```editorconfig
sudo apt-get update
sudo apt-get upgrade
```

Next, we will install some basic packages which we might need during the installation process as well in future. Also, remove the packages which are not needed.

```editorconfig
sudo apt-get install -y build-essential cmake gfortran git pkg-config
sudo apt-get install -y python-dev software-properties-common wget vim
sudo apt-get autoremove
```

### Step 2 : Install CUDA

**CUDA ( Compute Unified Device Architecture )** is a parallel computing platform and API developed by NVIDIA which utilizes the parallel computing capabilities of the GPUs. In order to use the graphics card, we need to have CUDA drivers installed on our system.

*If you do not have a NVIDIA CUDA supported Graphics Card, then you can skip this step. and go to Step 4.*

Download the CUDA driver from the [official nvidia website](https://developer.nvidia.com/cuda-downloads). We recommend you download the deb ( local ) version from Installer type as shown in the screenshot below.

![Cuda Download](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/cuda_download_01.png)

![Cuda Download](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/cuda_download_02.png)

After downloading the file, go to the folder where you have downloaded the file and run the following commands from the terminal to install the CUDA drivers.

Please make sure that the filename used in the command below is the same as the downloaded file.

```editorconfig
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

Run the following command to check whether the driver has installed successfully by running NVIDIA’s System Management Interface (nvidia-smi). It is a tool used for monitoring the state of the GPU.

```editorconfig
nvidia-smi
```

You should get an output as shown below.

![nvidia-smi](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/nvidia_smi.png)

### Step 3 : Install cuDNN

**CUDA Deep Neural Network (cuDNN)** is a library used for further optimizing neural network computations. It is written using the CUDA API.

Go to [official cuDNN website](https://developer.nvidia.com/rdp/form/cudnn-download-survey) and fill out the form for downloading the cuDNN library. After you get to the download link ( sample shown below ), you should download the “**cuDNN Library for Linux**” from the options.

![cuDNN download](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/cudnn_download.png)

Now, go to the folder where you have downloaded the “.tgz” file and from the command line execute the following.

```editorconfig
tar xvf cudnn-10.1-linux-x64-v7.6.0.64.tgz
sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/* /usr/local/cuda/include/
```

Next, update the paths for CUDA library and executables.

```editorconfig
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4 : Install requirements for DL Frameworks

Install dependencies of Deep Learning Frameworks

```editorconfig
sudo apt-get update
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev
```

Next, we install python 2 and 3 along with other important packages like boost, lmdb, glog, blas etc.

```editorconfig
sudo apt-get install -y --no-install-recommends libboost-all-dev doxygen
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev libblas-dev
sudo apt-get install -y libatlas-base-dev libopenblas-dev libgphoto2-dev libeigen3-dev libhdf5-dev

sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy
sudo apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy
```

### Step 5 : Enable Virtual Environments

In a virtual environment, you can install any python library without affecting the global installation or other virtual environments. This way, even if you damage the libraries in one virtual environment, your rest of the projects remain safe. It is highly recommended to use virtual environments.

Install the virtual environment wrapper which enables us to create and work on virtual environments in python.

```editorconfig
sudo pip2 install virtualenv virtualenvwrapper
sudo pip3 install virtualenv virtualenvwrapper

echo "# Virtual Environment Wrapper"  >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
```

### Step 6 : Install Deep Learning frameworks

Now, we install Tensorflow, Keras, PyTorch, dlib along with other standard Python ML libraries like numpy, scipy, sklearn etc.
We will create virtual environments (for python 3) and install all the deep learning frameworks inside them.

```editorconfig
# create a virtual environment for python 3
mkvirtualenv venv-dl -p python3
# Activate the virtual environment
workon venv-dl

pip install numpy scipy matplotlib scikit-image scikit-learn ipython protobuf jupyter

pip install tensorflow
# If you have CUDA installed
pip install tensorflow-gpu

pip install keras

pip3 install torch torchvision

pip install dlib

pip install opencv-python
pip install opencv-contrib-python

deactivate
```

Check Installation of Frameworks

```editorconfig
workon venv-dl
python
```

```python
import numpy
numpy.__version__
import tensorflow
tensorflow.__version__
import keras
keras.__version__
import torch
torch.__version__
import cv2
cv2.__version__
```

You should get an output similar to the figure shown below

![Test Installation](https://github.com/shin7/Notes/blob/master/Installing_Deep_Learning_Frameworks_with_CUDA_support/image/test_installation.png)

Cheer!!!

***

##### References

- https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/