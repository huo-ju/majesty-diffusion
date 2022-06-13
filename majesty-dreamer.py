from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus import AutoLockRenewer
from minio import Minio
from urllib.request import Request, urlopen

import torch
import majesty as majesty
from omegaconf import OmegaConf

import gc
import glob
import os

token = os.environ["NIGHTMAREBOT_WORKER_TOKEN"]
queue_name = os.environ["NIGHTMAREBOT_QUEUE_NAME"]
session_id = "test"
reply_session_id = "test-reply"
minio_key = os.environ["NIGHTMAREBOT_MINIO_KEY"]
minio_secret = os.environ["NIGHTMAREBOT_MINIO_SECRET"]

lock_renewal = AutoLockRenewer(max_workers=4)

# Get connection string from token - TODO flesh this out
connstr = (
    urlopen(
        Request(
            f"https://dreamer.nightmarebot.com/{token}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
    )
    .read()
    .decode("utf-8")
)

# Init
majesty.download_models()
torch.backends.cudnn.benchmark = True
majesty.device = torch.device("cuda")
config = OmegaConf.load(
    "./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
)
latent_diffusion_model = "finetuned"
model = majesty.load_model_from_config(
    config,
    f"{majesty.model_path}/latent_diffusion_txt2img_f8_large.ckpt",
    False,
    "latent_diffusion_model",
)
majesty.model = model.half().eval().to(majesty.device)
majesty.load_lpips_model()
majesty.load_aesthetic_model()
torch.cuda.empty_cache()
gc.collect()
majesty.opt.outdir = majesty.outputs_path

client = Minio(
    "dumb.dev",
    access_key=minio_key,
    secret_key=minio_secret,
)


def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
    assert os.path.isdir(local_path)

    for local_file in glob.glob(local_path + "/**"):
        local_file = local_file.replace(os.sep, "/")  # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(
                local_file,
                bucket_name,
                minio_path + "/" + os.path.basename(local_file),
            )
        else:
            content_type = "application/octet-stream"
            if local_file.endswith("png"):
                content_type = "image/png"
            if local_file.endswith("mp4"):
                content_type = "video/mp4"
            if local_file.endswith("jpg"):
                content_type = "image/jpg"
            remote_path = os.path.join(minio_path, local_file[1 + len(local_path) :])
            remote_path = remote_path.replace(
                os.sep, "/"
            )  # Replace \ with / on Windows
            client.fput_object(
                bucket_name, remote_path, local_file, content_type=content_type
            )


def process(id):
    workdir = os.path.join("/tmp", id)
    majesty.outputs_path = workdir
    majesty.custom_settings = f"{workdir}/settings.cfg"
    os.makedirs(workdir)
    client.fget_object(
        "nightmarebot-workflow", f"{id}/settings.cfg", majesty.custom_settings
    )
    majesty.load_custom_settings()
    majesty.full_clip_load()
    majesty.config_init_image()

    majesty.prompts = majesty.clip_prompts
    if majesty.latent_prompts == [] or majesty.latent_prompts == None:
        majesty.opt.prompt = majesty.prompts
    else:
        majesty.opt.prompt = majesty.latent_prompts
    majesty.opt.uc = majesty.latent_negatives
    majesty.set_custom_schedules()

    majesty.config_clip_guidance()
    majesty.config_output_size()
    majesty.config_options()

    torch.cuda.empty_cache()
    gc.collect()
    majesty.do_run()
    upload_local_directory_to_minio(workdir, "nightmarebot-output", id)


# Main loop
with ServiceBusClient.from_connection_string(connstr) as client:
    with client.get_queue_receiver(queue_name, session_id=session_id) as receiver:
        session = receiver.session
        lock_renewal.register(receiver, session, max_lock_renewal_duration=300)
        for message in receiver:
            process(str(id))
            with client.get_queue_sender(queue_name) as sender:
                sender.send_messages(
                    ServiceBusMessage(str(id), session_id=reply_session_id)
                )
            receiver.complete_message(message)
