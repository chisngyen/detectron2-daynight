# Sử dụng Python 3.8 base image
FROM python:3.8-slim

# Set timezone để tránh prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Cài đặt các system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-opencv \
    libglib2.0-0 \
    python3-pip \
    libgl1-mesa-glx \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Tạo các thư mục cần thiết
RUN mkdir -p /app/input /app/output /app/models

# Cài đặt PyTorch và các dependencies cơ bản
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
RUN pip install pyyaml opencv-python numpy

# Clone và cài đặt Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install -e detectron2

# Copy requirements và cài đặt các dependencies khác
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy model weights và code
COPY models/best_day.pth /app/models/best_day.pth
COPY models/best_night.pth /app/models/best_night.pth
COPY inference.py /app/inference.py

# Command mặc định khi chạy container
CMD ["python", "inference.py"]