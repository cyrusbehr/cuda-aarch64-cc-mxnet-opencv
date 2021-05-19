FROM cyrusbehr/aarch64-cuda10-2

COPY scripts/build_mxnet.sh / 
COPY patch/op.h /

RUN ./build_mxnet.sh