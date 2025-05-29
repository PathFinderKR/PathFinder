FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
LABEL authors="pathfinder"

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

# Install Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -u -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Create conda environment, install Jupyter Notebook and PyTorch
RUN conda create -y -n torch-env python=3.12 && \
    conda run -n torch-env pip install --upgrade pip && \
    conda run -n torch-env pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n torch-env pip install notebook && \
    conda run -n torch-env pip install -r requirements.txt

# Set the default conda environment to activate
CMD bash -c "\
    source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate torch-env && \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & \
    /usr/sbin/sshd -D"