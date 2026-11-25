FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg colmap git libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements*.txt ./
RUN python3 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && python3 -m pip install -r requirements.txt
COPY backend backend
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
ARG INSTALL_KERAS=false
RUN if [ "$INSTALL_KERAS" = "true" ]; then python3 -m pip install -r requirements-keras.txt; fi
ENV DATA_DIR=/data
EXPOSE 8000
CMD ["python3","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]
