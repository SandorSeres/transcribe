# NVIDIA alapú Python képfájl használata GPU támogatással
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python telepítése
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev gcc git curl libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


