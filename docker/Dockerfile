ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0"

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN git checkout main
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install -U openmim
RUN mim install mmengine
RUN pip install "mmdet<3.3"
RUN mim install "mmcv<2.2.0"
RUN mim install "mmpose>=1.1.0"
RUN mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
RUN python demo/inferencer_demo.py tests/data/coco/000000000785.jpg --pose3d human3d --vis-out-dir vis_results/human3d --pred-out-dir pred_results/human3d
RUN python demo/inferencer_demo.py tests/data/coco/000000000785.jpg --pose2d wholebody --vis-out-dir vis_results/wholebody --pred-out-dir pred_results/wholebody
RUN python demo/inferencer_demo.py tests/data/coco/000000000785.jpg --pose2d rtmo --vis-out-dir vis_results/rtmo --pred-out-dir pred_results/rtmo
RUN pip install flask
RUN ls
RUN cd /mmpose/projects/rtmpose3d/demo
RUN apt update && apt install wget -y
RUN wget https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
RUN mv /mmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth /mmpose/projects/rtmpose3d/rtmw3d-l_cock14-0d4ad840_20240422.pth
RUN mv /mmpose/projects/rtmpose3d/demo/* /mmpose/projects/rtmpose3d
RUN ls
COPY entrypoint.sh /mmpose/projects/rtmpose3d/entrypoint.sh
COPY start.sh /mmpose/projects/rtmpose3d/start.sh
COPY entrypoint.py /mmpose/projects/rtmpose3d/entrypoint.py
RUN chmod +x /mmpose/projects/rtmpose3d/entrypoint.sh
RUN chmod +x /mmpose/projects/rtmpose3d/entrypoint.py
ENTRYPOINT ["/mmpose/projects/rtmpose3d/entrypoint.sh"]