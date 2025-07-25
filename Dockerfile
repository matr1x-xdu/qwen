# syntax=docker/dockerfile:1
FROM m.daocloud.io/docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

# 设置环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR=/opt/conda

# 创建工作目录
RUN mkdir -p /home/admin/predict
WORKDIR /home/admin/predict

# 复制项目文件
COPY . /home/admin/predict

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    ln -s $CONDA_DIR/bin/conda /usr/bin/conda

RUN conda create -n atec2025 python=3.11 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    /bin/bash -c " \
    source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate atec2025 && \
    pip install --no-cache-dir -r requirements.txt" && \
    conda clean -y --all

# 设置环境变量
ENV PATH $CONDA_DIR/envs/atec2025/bin:$PATH

# COPY --from=model / /home/admin/predict/user-model-v3

# 验证安装
RUN python --version && \
    pip --version && \
    echo "Python path: $(which python)" && \
    echo "Pip path: $(which pip)" && \
    nvcc --version

# RUN modelscope login --token ms-47b5fc33-8812-4c10-90f6-960341a39857
# RUN modelscope download --model smileboy036/ATEC-2025-Qwen-Base --local_dir ./user-model-v3

# 设置入口点
RUN chmod +x /home/admin/predict/run.sh
ENTRYPOINT ["/home/admin/predict/run.sh"]
