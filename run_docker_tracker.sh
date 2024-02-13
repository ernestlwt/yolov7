docker run \
    --rm -it --gpus all \
    --shm-size=64g \
    --privileged \
    -v /home/ernestlwt/workspace/github/pp_eval/data:/data \
    -v /home/ernestlwt/workspace/github/yolov7:/workspace/yolo \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -v /dev:/dev\
    -e DISPLAY=$DISPLAY \
    ernestlwt/yolov7:bytetrack bash