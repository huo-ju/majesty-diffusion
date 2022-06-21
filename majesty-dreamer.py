from time import sleep
from azure.servicebus import ServiceBusClient, NEXT_AVAILABLE_SESSION
from azure.servicebus import AutoLockRenewer
from click import prompt
from minio import Minio
from urllib.request import Request, urlopen
import majesty as majesty
from subprocess import CalledProcessError, Popen, PIPE
import requests
import json
import datetime

import torch

from omegaconf import OmegaConf

import gc
import glob
import os

token = os.environ["NIGHTMAREBOT_WORKER_TOKEN"]
queue_name = os.environ["NIGHTMAREBOT_QUEUE_NAME"]
minio_key = os.environ["NIGHTMAREBOT_MINIO_KEY"]
minio_secret = os.environ["NIGHTMAREBOT_MINIO_SECRET"]
worker_env = os.getenv("NIGHTMAREBOT_WORKER_ENV")
worker_id = os.getenv("NIGHTMAREBOT_WORKER_ID")

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


clip_load_list = json.loads(
    urlopen(
        Request(
            f"https://dreamer.nightmarebot.com/clip_load_list",
            headers={"User-Agent": "Mozilla/5.0"},
        )
    )
    .read()
    .decode("utf-8")
)

minio_client = Minio(
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
            minio_client.fput_object(
                bucket_name, remote_path, local_file, content_type=content_type
            )


def process(id: str):
    print(f"processing {id}")
    workdir = os.path.join("/tmp", id)
    outdir = os.path.join(workdir, "results")
    custom_settings = f"{workdir}/settings.cfg"
    try:
        os.makedirs(outdir)
    except:
        if len(glob.glob(outdir + "/*.png")) > 0:
            print("output exists, skipping")
            return
    minio_client.fget_object(
        "nightmarebot-workflow", f"{id}/settings.cfg", custom_settings
    )

    with Popen(
        [
            "python",
            "latent.py",
            "-m",
            "/root/.cache/majesty",
            "-o",
            outdir,
            "-c",
            custom_settings,
            "--model_source",
            "https://storage.googleapis.com/majesty-diffusion-models",
        ],
        stdout=PIPE,
        bufsize=1,
        universal_newlines=True,
    ) as p:
        for line in p.stdout:
            print(line, end="")

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)

    images = glob.glob(outdir + "/*.png")
    minio_client.fput_object(
        "nightmarebot-output", f"{id}/{id}.png", images[0], content_type="image/png"
    )

    response = requests.post(
        f"https://nightmarebot.azurewebsites.net/api/ProcessResult?token={token}&id={id}"
    )
    print(response.headers)
    # upload_local_directory_to_minio(workdir, "nightmarebot-output", id)


majesty.model_path = "/root/.cache/majesty"
majesty.model_source = "http://models.nmb.ai/majesty"
import models

models.download_models(majesty.model_path, majesty.model_source, erlich=True, ongo=True)
models.download_clip(clip_load_list=clip_load_list)

# Run for 5 minutes without messages then request shutdown
runUntil = datetime.datetime.now() + datetime.timedelta(minutes=1)
with ServiceBusClient.from_connection_string(connstr) as client:
    while not worker_id or datetime.datetime.now() < runUntil:
        try:
            session_id = os.getenv("NIGHTMAREBOT_SESSION_ID")
            wait_time = 60 * 5
            if session_id == "" or session_id == None:
                session_id = NEXT_AVAILABLE_SESSION
                wait_time = 5
            with client.get_queue_receiver(
                queue_name, session_id=session_id, max_wait_time=wait_time
            ) as receiver:
                session = receiver.session
                print(f"polling session {session.session_id}\n", flush=True)
                for message in receiver:
                    lock_renewal.register(receiver, session)
                    request_id: str = str(message)
                    try:
                        process(request_id)
                    except Exception as e:
                        print(
                            f"Error processing request:{e}",
                            flush=True,
                        )
                    try:
                        runUntil = datetime.datetime.now() + datetime.timedelta(
                            minutes=1
                        )
                        receiver.complete_message(message)
                    except Exception as e:
                        print(f"Error completing message:{e}", flush=True)
                        try:
                            client.get_queue_receiver(
                                queue_name, session_id=session_id
                            ).complete_message(message)
                        except Exception as e:
                            print(f"Completion retry failed: {e}")

        except Exception as e:
            print(f"Listen failed: {e}", flush=True)
            sleep(5)

while True:
    print("idle, requesting shutdown\n", flush=True)
    response = requests.post(
        f"https://nightmarebot.azurewebsites.net/api/IdleWorker?token={token}&env={worker_env}&id={worker_id}"
    )
    sleep(60)