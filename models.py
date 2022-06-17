import mmc
from mmc.registry import REGISTRY
import mmc.loaders  # force trigger model registrations
from mmc.mock.openai import MockOpenaiClip
import os, shutil, subprocess
import argparse


def download_models(
    model_path=os.path.expanduser("~/.cache/majesty"),
    model_source="http://models.nmb.ai/majesty",
    ongo=False,
    erlich=False,
):
    # download models as needed
    models = [
        [
            "latent_diffusion_txt2img_f8_large.ckpt",
            "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt",
        ],
        [
            "txt2img-f8-large-jack000-finetuned-fp16.ckpt",
            "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-jack000-finetuned-fp16.ckpt",
        ],
        [
            "ava_vit_l_14_336_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_336_linear.pth",
        ],
        [
            "sa_0_4_vit_l_14_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_l_14_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_b_16_linear.pth",
            "http://batbot.tv/ai/models/v-diffusion/ava_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_16_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_32_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_32_linear.pth",
        ],
        [
            "openimages_512x_png_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/openimages_512x_png_embed224.npz",
        ],
        [
            "imagenet_512x_jpg_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/imagenet_512x_jpg_embed224.npz",
        ],
        [
            "GFPGANv1.3.pth",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        ],
    ]

    if ongo:
        models.append(
            [
                "ongo.pt",
                "https://huggingface.co/laion/ongo/resolve/main/ongo.pt",
            ]
        )
    if erlich:
        models.append(
            [
                "erlich.pt",
                "https://huggingface.co/laion/erlich/resolve/main/model/ema_0.9999_120000.pt",
            ]
        )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for item in models:
        model_file = f"{model_path}/{item[0]}"
        if not os.path.exists(model_file):
            if model_source:
                url = f"{model_source}/{item[0]}"
            else:
                url = item[1]
            print(f"Downloading {url}", flush=True)
            subprocess.call(
                ["wget", "-nv", "-O", model_file, url, "--no-check-certificate"],
                shell=False,
            )
    if not os.path.exists("GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
        shutil.copyfile(
            f"{model_path}/GFPGANv1.3.pth",
            "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
        )


def get_mmc_models(clip_load_list):
    mmc_models = []
    for model_key in clip_load_list:
        if not model_key:
            continue
        arch, pub, m_id = model_key[1:-1].split(" - ")
        mmc_models.append(
            {
                "architecture": arch,
                "publisher": pub,
                "id": m_id,
            }
        )
    return mmc_models


def download_clip(clip_load_list):
    # Download CLIP models
    for item in get_mmc_models(clip_load_list):
        clip_loaders = REGISTRY.find(**item)
        for loader in clip_loaders:
            print(loader)
            try:
                model = loader.load(clip_load_list)
                del model
            except:
                print("Ignoring load error\n")


def main():

    parser = argparse.ArgumentParser(
        description="Download models for majesty diffusion"
    )
    parser.add_argument(
        "--erlich", help="Download erlich model", dest="erlich", action="store_true"
    )
    parser.add_argument(
        "--ongo", help="Download ongo model", dest="ongo", action="store_true"
    )

    parser.add_argument(
        "--clip",
        metavar="M",
        nargs="+",
        type=str,
        help="CLIP model(s) to load",
        dest="clip_load_list",
    )

    args = parser.parse_args()

    if args.clip_load_list:
        download_clip(args.clip_load_list)
    else:
        download_models(ongo=args.ongo, erlich=args.erlich)


if __name__ == "__main__":
    main()
