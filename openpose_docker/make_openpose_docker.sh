#!/bin/bash
# download all essential package

# Download models for our algorithmn running, you need to install and connect to your aws before running
echo "================================> Copy model from s3 <============================================="
aws s3 cp s3://aistetic-db-models/Openpose_model ./model --recursive

# Run docker file to build image
echo "================================> Build openpose docker <============================================="
docker build -t openpose ./

