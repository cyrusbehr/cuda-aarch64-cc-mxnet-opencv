FROM aarch64-cuda10-2

COPY scripts/build_opencv.sh / 

RUN ./build_opencv.sh