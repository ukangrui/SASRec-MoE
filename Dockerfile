FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /moelora/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git
RUN pip install accelerate==0.32.1
RUN pip install hydra-core==1.3.2
RUN pip install numpy==1.26.4
RUN pip install omegaconf==2.3.0
RUN pip install pandas==1.5.3
RUN pip install git+https://github.com/BenjaminBossan/peft.git@d1f6ab2ede1417e6a852eb85da95ab2c72f27a3a
RUN pip install pytorch-lightning==1.9.4
RUN pip install PyYAML==6.0.1
RUN pip install scikit-learn==1.5.0
RUN pip install scipy==1.13.1
RUN pip install torch==2.3.0