[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# How to Cross Compile OpenCV and MXNET for AArch64 CUDA (NVIDIA Jetson)

## Overview
The following project demonstrates how you can cross compile OpenCV and MXNET for AArch64 CUDA. The resulting libraries can be used on an NVIDIA Jetson. 

## Prerequisites
You must have docker installed.

## Build instructions
- `docker build -t cyrusbehr/aarch64-cuda10-2 -f aarch64-cuda-10-2-ubuntu18.dockerfile .`
- `docker build -t cyrusbehr/opencv-cuda -f build-opencv-cuda.dockerfile .`


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/cuda-aarch64-cc-mxnet-opencv.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/cuda-aarch64-cc-mxnet-opencv/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/cuda-aarch64-cc-mxnet-opencv.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/cuda-aarch64-cc-mxnet-opencv/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/
