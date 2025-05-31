# Prerequisites:
# 1. Unraid Server: This solution is designed for Unraid OS.
# 2. NVIDIA GPU: An NVIDIA graphics card compatible with CUDA is required.
# 3. NVIDIA GPU Drivers (on Unraid): Ensure up-to-date NVIDIA drivers are installed on your Unraid server.
# 4. NVIDIA Container Toolkit (on Unraid): This is crucial for Docker to access the GPU. Verify that "NVIDIA Container Toolkit" or "GPU Support" is enabled in Unraid's Docker settings.
# 5. Sufficient Disk Space:
#    - For Docker Image: The Docker image (with libraries and models) can be several gigabytes. Ensure enough free space on the drive/cache drive where Docker stores its images.
#    - For Video Files and SRTs: Ensure sufficient free space on the share where your video files are located and where the generated SRT files will be saved.
# 6. Internet Connection (during build and initial run): Required for downloading the base image, Python packages, and Hugging Face models.
# 7. SSH access or Terminal in Unraid: Command-line access is needed to build the Docker image and start containers.

# Use the official CUDA base image from NVIDIA.
# This ensures that the necessary CUDA drivers and libraries are present.
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Ensure that the PATH is correctly set for pip executables
ENV PATH="/usr/local/bin:$PATH"

# Update apt-get and install necessary system software.
# ffmpeg is needed for audio processing.
# git is often needed for downloading repositories.
# python3-pip is needed to install Python packages.
# libcudnn8 is for GPU acceleration with CUDA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3-pip \
    libcudnn8 libcudnn8-dev

# Set the working directory in the container.
# All operations, such as mounting volumes, will be relative to this directory.
WORKDIR /data

# Install Python packages with pip.
# --no-cache-dir to reduce the installation cache in the final image.

# ctranslate2 is a dependency of WhisperX, specific version for compatibility.
RUN pip3 install --no-cache-dir "ctranslate2==3.18.0"

# Install WhisperX with CUDA support.
# The '~=3.3.0' ensures compatibility within the 3.3.x series.
RUN pip3 install --no-cache-dir "whisperx[cuda]~=3.3.0"

# Install the Hugging Face Transformers library and pysrt.
# torch is the deep learning framework, with the correct CUDA version installed by the base image.
# transformers is the library for loading and using models like MarianMT.
# sentencepiece is a dependency of many Hugging Face tokenizers.
# pysrt is for working with SRT files.
# sacremoses is a recommended dependency for MarianMT tokenizers.
RUN pip3 install --no-cache-dir \
    "torch==2.7.0" \
    "transformers==4.52.4" \
    "torchaudio==2.7.0" \
    "sentencepiece" \
    "pysrt" \
    "sacremoses"

# Define an argument for the target language with a default value (e.g., nl)
ARG TARGET_LANG_DEFAULT="nl"
ENV TARGET_LANGUAGE=${TARGET_LANG_DEFAULT}

# Copy your custom Python script to a location in the container's PATH.
COPY createSrt.py /usr/local/bin/createSrt.py

# Make the script executable.
RUN chmod +x /usr/local/bin/createSrt.py

# Define the entrypoint of the container.
# This Python script will be executed when the container starts.
ENTRYPOINT ["/usr/local/bin/createSrt.py"]
