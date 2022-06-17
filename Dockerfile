FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential wget git git-lfs \
    ffmpeg libsm6 libxext6 \
    && apt-get clean

RUN mkdir -p /root/.cache/majesty
WORKDIR /root/.cache/majesty
RUN wget https://models.nmb.ai/majesty/latent_diffusion_txt2img_f8_large.ckpt
RUN wget https://models.nmb.ai/majesty/txt2img-f8-large-jack000-finetuned-fp16.ckpt
RUN wget https://models.nmb.ai/majesty/GFPGANv1.3.pth
RUN wget https://models.nmb.ai/majesty/ava_vit_b_16_linear.pth
RUN wget https://models.nmb.ai/majesty/ava_vit_l_14_336_linear.pth
RUN wget https://models.nmb.ai/majesty/ava_vit_l_14_linear.pth
RUN wget https://models.nmb.ai/majesty/imagenet_512x_jpg_embed224.npz
RUN wget https://models.nmb.ai/majesty/openimages_512x_png_embed224.npz
RUN wget https://models.nmb.ai/majesty/sa_0_4_vit_b_16_linear.pth
RUN wget https://models.nmb.ai/majesty/sa_0_4_vit_b_32_linear.pth
RUN wget https://models.nmb.ai/majesty/sa_0_4_vit_l_14_linear.pth
RUN wget https://models.nmb.ai/majesty/erlich.pt
RUN wget https://models.nmb.ai/majesty/ongo.pt

RUN mkdir -p /root/.cache/clip
RUN ln -s /src/models /root/.cache/majesty
WORKDIR /root/.cache/clip
RUN wget https://models.nmb.ai/clip/RN50x16.pt
RUN wget https://models.nmb.ai/clip/RN50x4.pt
RUN wget https://models.nmb.ai/clip/RN50x64.pt
RUN wget https://models.nmb.ai/clip/ViT-B-16.pt
RUN wget https://models.nmb.ai/clip/ViT-B-32.pt
RUN wget https://models.nmb.ai/clip/ViT-L-14-336px.pt
RUN wget https://models.nmb.ai/clip/ViT-L-14.pt
RUN wget https://models.nmb.ai/clip/vit_b_16_plus_240-laion400m_e32-699c4b84.pt
RUN wget https://models.nmb.ai/clip/vit_b_32-laion2b_e16-af8dbd0c.pth

RUN mkdir -p /content
RUN ln -s /root/.cache/majesty /content/models
WORKDIR /content

RUN git clone https://github.com/multimodalart/latent-diffusion --branch 1.4
RUN git clone https://github.com/CompVis/taming-transformers
RUN git clone https://github.com/TencentARC/GFPGAN
RUN git lfs clone https://github.com/LAION-AI/aesthetic-predictor

RUN pip install tensorflow==2.9.1
RUN pip install -e ./taming-transformers
RUN pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
RUN pip install transformers
RUN pip install dotmap
RUN pip install resize-right
RUN pip install piq
RUN pip install lpips
RUN pip install basicsr
RUN pip install facexlib
RUN pip install realesrgan
RUN pip install ipywidgets
RUN pip install opencv-python
RUN pip install jupyterlab
RUN pip install --upgrade jupyter_http_over_ws>=0.0.7 && \
    jupyter serverextension enable --py jupyter_http_over_ws && \
    jupyter nbextension enable --py widgetsnbextension

RUN git clone https://github.com/apolinario/Multi-Modal-Comparators --branch gradient_checkpointing
RUN pip install poetry
WORKDIR /content/Multi-Modal-Comparators
RUN poetry build; pip install dist/mmc*.whl
WORKDIR /content
RUN python Multi-Modal-Comparators/src/mmc/napm_installs/__init__.py

RUN git lfs clone https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings

COPY *.py ./
COPY *.ipynb ./

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "latent.ipynb", "--NotebookApp.allow_origin", "https://colab.research.google.com", "--NotebookApp.port", "8888", "--NotebookApp.ip", "*", "--NotebookApp.port_retries", "0", "--allow-root"]