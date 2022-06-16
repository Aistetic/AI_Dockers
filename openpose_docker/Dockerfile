FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub


RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev

# Replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

# install all python requirements
# depress the warning
RUN python3 -m pip install pip==21.0.1

WORKDIR /app
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt

########################################################## 1. Setup Openpose here ###############################################
# Install openpose, it crash on cudnn8
WORKDIR /app/openpose
RUN git clone -b v1.7.0 https://github.com/dizhongzhu/openpose.git .

# Build it
WORKDIR /app/openpose/build
RUN cmake -DBUILD_PYTHON=ON -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake
RUN make -j `nproc`

##################################################  Copy models ########################################################
WORKDIR /app/model/Openpose_model
COPY ./model ./

########################################## Set PYTHONPATH ###############################################
ENV PYTHONPATH=/app:/app/openpose/build/python:$PYTHONPATH

# Add listener for the image
WORKDIR /app/src
COPY ./src .

# Run openpose cmd
ENTRYPOINT ["python3","-u","openpose_app.py"]