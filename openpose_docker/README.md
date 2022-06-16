# Compile an openpose docker image that for you convinient to use

Run the 'make_openpose_docker.sh' script to compile the docker image. If you want to use the pre-trained model when no access to AWS S3. 
Please edit the Dockerfile in line
```
RUN cmake -DBUILD_PYTHON=ON -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
```
to
```
RUN cmake -DBUILD_PYTHON=ON -DDOWNLOAD_BODY_25_MODEL=ON -DDOWNLOAD_FACE_MODEL=ON -DDOWNLOAD_HAND_MODEL=ON ..
```

### Download the docker image directly from Docker Hub

You can pull the compiled docker image by
```angular2html
docker pull dizhongzhu/openpose:latest
```

Please refer the [openpose flag](https://hub.docker.com/r/humanbodyreconstruction/openpose/) to see what flag you need to set in order to run the pose detection by your needs.

