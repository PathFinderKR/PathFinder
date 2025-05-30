FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
LABEL authors="jwkim" \
      description="PyTorch development environment with CUDA 12.1 and cuDNN 9"

# Instal system packages
RUN apt-get update && apt-get install -y \
    wget \
    net-tools \
    openssh-server \
    build-essential \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Optional: root password (default: 'password')
ARG ROOT_PASSWORD=password

# SSH configuration
RUN mkdir /var/run/sshd && \
    echo "root:${ROOT_PASSWORD}" | chpasswd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config && \
    echo "UseDNS no" >> /etc/ssh/sshd_config
EXPOSE 22 8888

# Install cuDNN
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.10.1/local_installers/cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.10.1_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.10.1/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cudnn

# Install Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -u -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Create conda environment, install Jupyter Notebook and PyTorch
RUN conda create -y -n torch-env python=3.12 && \
    conda run -n torch-env pip install --upgrade pip && \
    conda run -n torch-env pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n torch-env pip install notebook

# Workspace setup
WORKDIR /workspace

# Default command to start SSH server
CMD ["/usr/sbin/sshd", "-D"]

# To build the Docker image:
# docker build -t jwkim/pytorch --build-arg ROOT_PASSWORD=password .

# To run the Docker container:
# docker run -d --gpus all --shm-size=8G --restart always --name jwkim_projects -p 20022:22 -p 20088:8888 -v ~/projects:/workspace jwkim/pytorch

# Conda setup in the container:
# echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
# echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
# echo 'conda activate torch-env' >> ~/.bashrc
# source ~/.bashrc
# conda init
