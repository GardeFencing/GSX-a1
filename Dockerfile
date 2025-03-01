FROM ubuntu:22.04

# Install apt dependencies
RUN apt update && apt install -y \
    build-essential \
    cmake \
    pkg-config \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    ccache \
    wget \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Clone llama.cpp and build it
RUN git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git /app/llama.cpp
WORKDIR /app/llama.cpp
RUN make LLAMA_FFMPEG=1

# Copy your source code
WORKDIR /app
COPY . /app/

# Start your app
CMD ["/app/docker-setup.sh"]
