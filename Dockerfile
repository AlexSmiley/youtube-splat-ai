FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip build-essential cmake ninja-build ffmpeg colmap git libgl1 libegl1 libgl1-mesa-dri xvfb xauth && rm -rf /var/lib/apt/lists/*
ENV QT_QPA_PLATFORM=offscreen
WORKDIR /app
COPY requirements*.txt ./
RUN python3 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install -r requirements.txt
# Nerfstudio imports OpenCV before launching COLMAP; isolate system Qt.
RUN mv /usr/bin/colmap /usr/bin/colmap.real && printf '%s\n' '#!/bin/sh' 'unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR' 'export QT_QPA_PLATFORM=offscreen' 'exec /usr/bin/colmap.real "$@"' > /usr/local/bin/colmap && chmod +x /usr/local/bin/colmap
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
ARG INSTALL_KERAS=false
RUN if [ "$INSTALL_KERAS" = "true" ]; then python3 -m pip install -r requirements-keras.txt; fi
COPY backend backend
ENV DATA_DIR=/data
EXPOSE 8000
CMD ["python3","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]
