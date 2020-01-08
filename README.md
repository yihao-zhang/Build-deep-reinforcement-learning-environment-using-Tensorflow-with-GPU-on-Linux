# Build-deep-reinforcement-learning-environment-using-Tensorflow-with-GPU-on-Linux

Build deep reinforcement learning environment using Tensorflow with GPU on Linux

When I started working on deep reinforcement learning, the first issue I had is how to build a robust deep reinforcement learning environment. For users who work on a HPC cluster where you are not allowed to install anything, it is crucial to use the python libraries that are required but not installed in the system. An excellent approach is to use the Singularity container.
I will list the steps to build a deep reinforcement learning environment on Ubuntu 18.04/CentOS 7 system using the singularity container and run it on a linux HPC cluster. If you are working on a machine where you will have full control, the easiest way to build a deep reinforcement learning environment using Tensorflow GPU is to install the Anaconda. The singularity container is an environment that you can have full control. This is extremely useful when you are working on a HPC cluster since you don’t need to ask your cluster admin to install anything for you. Also, you can maintain python libraries of different versions inside a container without changing the system’s configuration. More information about the Singularity can be found at https://singularity.lbl.gov/

Prerequisite:
1. A Ubuntu 18.04/CentOS 7 system where you have full control. 
Note: This system is used to build the singularity container for the deep reinforcement learning environment. You will need to have the sudo right to build and modify the container as I will show later.
2. The latest NVIDIA driver is installed. 
For HPC cluster users, usually you don’t need to worry about the installation, the system admin takes care of it. 
If you want to run Tensorflow with GPU on you own machine, you will need to install the NVIDIA driver. 
For Ubuntu users, check out this article https://medium.com/@bbloks/a-machine-learning-environment-with-ubuntu-and-gpu-acceleration-in-5-steps-765608325356
For CentOS users, check out this article https://www.advancedclustering.com/act_kb/installing-nvidia-drivers-rhel-centos-7/ 

Below are step-by-step instructions to build a singularity container which includes: CentOS7, Tensorflow1.12, Python3.5, Cuda9.0, cuDNN7.4, and TRFL 1.0.

Step 1. Download and install the singularity on your machine.
wget https://github.com/singularityware/singularity/releases/download/2.5.1/singularity-2.5.1.tar.gz
tar -zxf singularity-2.5.1.tar.gz
cd singularity-2.5.1/
./configure 
Note: There may be some issues when your system does not satisfy the requirement, you will need to install the missing parts as mentioned in the error message such as gcc, make, libarchive-dev etc.
make
sudo make install
After the installation is finished
singularity --version 
To check if it is installed successfully.

Step 2. Create an empty Container.
singularity create -s 5120 tensorflow.img
Here I create an empty container with size of 5GB and named “tensorflow.img”. I recommend user to specify a size larger than 5120MB as shown here to ensure enough space for future potential installations in the container.

Step 3. Create the .def file which is used to bootstrap the image.
An example is shown in the cent.def file.
Other examples can be found in the links below:
https://public.confluence.arizona.edu/display/UAHPC/Singularity+-+CentOS7%2C+Theano0.9%2C+Python3.4%2C+Cuda7.5%2C+cuDNN5.1
https://public.confluence.arizona.edu/display/UAHPC/Singularity+-+CentOS7%2C+Tensorflow1.2%2C+Python3.5%2C+Cuda8.0%2C+cuDNN5.1
Note: in the cent.def, there is a line
“cp ./cudnn-9.0-linux-x64-v7.4.1.5.tgz $SINGULARITY_ROOTFS”
The “cudnn-9.0-linux-x64-v7.4.1.5.tgz” is the cuDNN v7.4.1 for CUDA 9.0 installation file which can be found at https://developer.nvidia.com/cudnn. You'll need to download it from NVIDIA and modify the path accordingly.

4. Install the yum
CentOS users may skip this step
sudo apt install yum  
Create a file named .rpmmacros which include the following lines
%_var /var
%_dbpath %{_var}/lib/rpm
Then sudo cp . rpmmacros /root/

5. Build the Singularity container
sudo singularity bootstrap tensorflow.img cent.def
After the container is built, you can try using the python3.5 and import tensorflow to check if it works.

6. Upgrade Tensorflow GPU
sudo singularity shell --writable tensor.img
pip3.5 install --upgrade tensorflow-gpu==1.12.0
This version of tensorflow gpu is compatible with CUDA and cuDNN installed

7. Install TRFL
TRFL is a library built on top of TensorFlow that exposes several useful building blocks for implementing Reinforcement Learning agents. https://github.com/deepmind/trfl
sudo singularity shell --writable tensor.img
pip3.5 install trfl
pip3.5 install --upgrade trfl==1.0	
pip3.5 install --upgrade tensorflow_probability==0.5	

Step 6 and 7 show how to modify the container by installing new libraries.  
This container can be uploaded to a HPC cluster and used to execute python files by submitting the jobs with GPU application. 
Command to execute Singularity container with GPU enabled
singularity exec --nv tensorflow.img python3.5 *.py 

For users working on their own machine.
1. Install NVIDIA driver
2. Install Anaconda or Minoconda
https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent
3. Build deep reinforcement learning environment
conda create -n tf-gpu python==3.5 tensorflow-gpu=1.9.0 
Here I create an environment named tf-gpu with python 3.5 and tensorflow-gpu version 1.9.0
conda activate tf-gpu
pip install --upgrade tensorflow-gpu==1.12.0
pip install trfl==1.0
pip install tensorflow-probability==0.5

