# Use pytorch/pytorch image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Persistent Volume
VOLUME /workspace

# Install necessary packages
RUN apt update && apt install -y \
    git \
    build-essential \
    gfortran \
    cmake \
    vim \
    meson \
    ninja-build \
    libgsl-dev \
    htop \
    wget \
    zlib1g-dev \
    libopenblas-openmp-dev \
    pkg-config \
    openssh-client \
    libopenblas-dev \
    python3-dev \
    autoconf \
    automake \
    libtool 

# Set environment
ENV GPU_ARCHITECTURE=Turing
ENV CONDA_PREFIX=/opt/conda
ENV Torch_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake
ENV OMP_NUM_THREADS=1
ENV OPT_DIR=/opt/gromos-torch-gpu
ENV SRC_DIR=${OPT_DIR}/src
ENV BIN_DIR=${OPT_DIR}/bin
RUN mkdir -p ${OPT_DIR} ${SRC_DIR} ${BIN_DIR}

# Files needed to patch PyTorch extensions and xtb
COPY patch /workspace/patch

# Clone and install PyTorch extensions
WORKDIR ${SRC_DIR}
RUN git clone --recursive https://github.com/rusty1s/pytorch_cluster.git --branch 1.6.3 --depth 1 && \
    cd pytorch_cluster && \
    patch CMakeLists.txt < /workspace/patch/torch_cluster.patch && \
    cmake -DTORCH_CUDA_ARCH_LIST=${GPU_ARCHITECTURE} -DWITH_CUDA=on -DCMAKE_INSTALL_PREFIX=${BIN_DIR}/torch-extensions -S . -B build && \
    cmake --build build -j20 && \
    cmake --install build

WORKDIR ${SRC_DIR}
RUN git clone --recursive https://github.com/rusty1s/pytorch_sparse.git --branch 0.6.18 --depth 1 && \
    cd pytorch_sparse && \
    patch CMakeLists.txt < /workspace/patch/torch_sparse.patch && \
    cmake -DTORCH_CUDA_ARCH_LIST=${GPU_ARCHITECTURE} -DWITH_CUDA=on -DCMAKE_INSTALL_PREFIX=${BIN_DIR}/torch-extensions -S . -B build && \
    cmake --build build -j20 && \
    cmake --install build

WORKDIR ${SRC_DIR}
RUN git clone --recursive https://github.com/rusty1s/pytorch_scatter.git --branch 2.1.2 --depth 1 && \
    cd pytorch_scatter && \
    patch CMakeLists.txt < /workspace/patch/torch_scatter.patch && \
    cmake -DTORCH_CUDA_ARCH_LIST=${GPU_ARCHITECTURE} -DWITH_CUDA=on -DCMAKE_INSTALL_PREFIX=${BIN_DIR}/torch-extensions -S . -B build && \
    cmake --build build -j20 && \
    cmake --install build

# Clone and install xtb
WORKDIR ${SRC_DIR}
RUN git clone --recursive --depth 1 https://github.com/grimme-lab/xtb.git --branch v6.5.1 && \
    cd xtb && \
    patch src/api/calculator.f90 < /workspace/patch/xtb_api.patch && \
    patch include/xtb.h < /workspace/patch/xtb_include.patch && \
    patch test/api/c_api_example.c < /workspace/patch/xtb_test.patch && \
    meson setup build --buildtype=release -Dla_backend=openblas && \
    meson configure build --prefix=${BIN_DIR}/xtb-6.5.1 && \
    ninja -C build -j20 && \
    ninja -C build test && \
    ninja -C build install

# Install OpenMPI
WORKDIR ${SRC_DIR}
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz && \
    tar -xvf openmpi-4.1.6.tar.gz && \
    cd openmpi-4.1.6 && \
    ./configure --enable-mpi-cxx --prefix ${BIN_DIR}/openmpi && \
    make -j20 && \
    make install

# MPI in path
ENV PATH="$PATH:${BIN_DIR}/openmpi/bin"
ENV LIBRARY_PATH="$LIBRARY_PATH:${BIN_DIR}/openmpi/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BIN_DIR}/openmpi/lib"

# Install fftw
WORKDIR ${SRC_DIR}
RUN wget http://www.fftw.org/fftw-3.3.10.tar.gz && \
    tar -xvf fftw-3.3.10.tar.gz && \
    cd fftw-3.3.10 && \
    ./configure --enable-mpi --enable-shared --prefix ${BIN_DIR}/fftw3 && \
    make -j20 && \
    make install

# Set up environment for gromos
ENV PKG_CONFIG_PATH="$PKG_CONFIG_PATH:${BIN_DIR}/xtb-6.5.1/lib/x86_64-linux-gnu/pkgconfig"
ENV LIBRARY_PATH="$LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:${BIN_DIR}/xtb-6.5.1/lib/x86_64-linux-gnu:${BIN_DIR}/torch-extensions/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${BIN_DIR}/fftw3/lib:${CONDA_PREFIX}/lib"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:${BIN_DIR}/xtb-6.5.1/lib/x86_64-linux-gnu:${BIN_DIR}/torch-extensions/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${BIN_DIR}/fftw3/lib:${CONDA_PREFIX}/lib"

# Clone and install gromos
WORKDIR ${SRC_DIR}
RUN git clone https://github.com/rinikerlab/gromosXX --branch torch --depth 1 && \
    cd gromosXX/gromosXX && \
    cmake -S . -B build -DMPI=on -DXTB=on -DTORCH=on -DTORCH_CUDA_ARCH_LIST=${GPU_ARCHITECTURE} -DCMAKE_PREFIX_PATH="${BIN_DIR}/torch-extensions;${BIN_DIR}/xtb-6.5.1;${BIN_DIR}/fftw3" -DCMAKE_INSTALL_PREFIX=${BIN_DIR}/gromosXX && \
    cmake --build build -j20 && \
    cmake --install build

# Add gromos to path
ENV PATH="$PATH:${BIN_DIR}/gromosXX/bin"

# Clone and install gromos
WORKDIR ${SRC_DIR}
RUN git clone https://github.com/rinikerlab/gromosPlsPls.git --branch torch --depth 1 && \
    cd gromosPlsPls/gromos++ && \
    ./Config.sh && \
    mkdir -p build && \
    cd build && \
    ../configure --disable-shared --disable-debug --enable-openmp --with-fftw=${BIN_DIR}/fftw3 --with-gsl=/usr/lib --prefix ${BIN_DIR}/gromos++ && \
    make -j20 && \
    make install

# Add gromos++ to path
ENV PATH="$PATH:${BIN_DIR}/gromos++/bin"

# Clone AMP and install requirements
WORKDIR /workspace
RUN git clone --recursive https://github.com/rinikerlab/amp_qmmm --depth 1 && \
    pip install torchlayers tensorboard && \
    conda install torchmetrics pytorch-scatter=2.1.2  pytorch-sparse=0.6.18 pytorch-cluster=1.6.3 -c pyg -c conda-forge

# Add user
RUN useradd -ms /bin/bash amp
RUN chown -R amp:amp /workspace
USER amp
