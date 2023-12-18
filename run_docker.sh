docker run \
    --rm -it --gpus all \
    --shm-size=64g \
    --privileged \
    -v /home/ernestlwt/data:/data \
    -v /home/ernestlwt/workspace/github/yolov7:/workspace/yolov7 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -v /dev:/dev\
    -e DISPLAY=$DISPLAY \
    yolov7 bash