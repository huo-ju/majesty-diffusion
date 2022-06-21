# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from subprocess import Popen, PIPE, CalledProcessError
import tempfile
import glob
import models
import majesty
import os,shutil
import typing

sizes = [128,192,256,320,384]
model_path = "/root/.cache/majesty-diffusion"
class Predictor(BasePredictor):
    def setup(self):        
        os.environ["TOKENIZERS_PARALELLISM"] = "true"
        # move preloaded clip models into cache
        print('Ensuring models are loaded..')
        models.download_models(model_path=model_path)
        #models.download_clip(majesty.clip_load_list)
        if not os.path.exists("/src/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
            shutil.copyfile(
                f"{model_path}/GFPGANv1.3.pth",
                "/src/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
        )

        
    def predict(
        self,
        clip_prompt: str = Input(description="Prompt for CLIP guidance", default="The portrait of a Majestic Princess, trending on artstation", max_length=230),
        latent_prompt: str = Input(description="Prompt for latent diffusion", default="The portrait of a Majestic Princess, trending on artstation", max_length=230),
        model: str = Input(description="Latent diffusion model", default='finetuned', choices=["original", "finetuned", "ongo", "erlich"]),
        latent_negatives: str = Input(description="Negative prompts for Latent Diffusion", default=None),
        height: int = Input(description="Output height (output will be scaled up 1.5x with default settings)", default=256, choices=sizes),
        width: int = Input(description="Output width (output will be scaled up 1.5x with default settings)", default=256, choices=sizes),
        init_image: Path = Input(description="Initial image", default=None),
        init_mask: Path = Input(description="A mask same width and height as the original image with the color black indicating where to inpaint", default=None),
        init_scale: int = Input(description="Controls how much the init image should influence the final result. Experiment with values around 1000", default=1000),
        init_brightness: float = Input(description="Init image brightness", default=0.0),
        latent_scale: int = Input(description="The `latent_diffusion_guidance_scale` will determine how much the `latent_prompts` affect the image. Lower help with text interpretation, higher help with composition. Try values between 0-15. If you see too much text, lower it", default=12),
        clip_scale: int = Input(description="CLIP guidance scale", default=16000),
        aesthetic_loss_scale: int = Input(description="Aesthetic loss scale", default=400),
        starting_timestep: float = Input(description="Starting timestep", default=0.9),
        num_batches: int = Input(description="Number of batches", default=1, ge=1, le=10),
        custom_settings: Path = Input(description="Advanced settings file", default=None),
    ) -> typing.List[Path]:
        """Run a single prediction on the model"""

        outdir = tempfile.mkdtemp('majesty')

        command = [
                "python",
                "latent.py",
                "--clip_prompt",
                clip_prompt,
                "--latent_prompt",
                latent_prompt,
                "--latent_diffusion_model",
                model,
                "--latent_scale",
                str(latent_scale),
                "--clip_scale",
                str(clip_scale),
                "--aesthetic_loss_scale",
                str(aesthetic_loss_scale),
                "--height",
                str(height),
                "--width",
                str(width),
                "--batches",
                str(num_batches),
                "--starting_timestep",
                str(starting_timestep),
                "-m",
                "/root/.cache/majesty-diffusion",
                "-o",
                outdir,
                "--model_source",
                "https://models.nmb.ai/majesty",
            ]
        if init_image:
            command.append(["--init_image", init_image])
            command.append(["--init_scale", init_scale])
            command.append(["--init_brightness", init_brightness])
            if init_mask:
                command.append(["--init_mask", init_mask])

        if latent_negatives:
            command.append(["--latent_negatives", latent_negatives])
        if custom_settings:
            command.append(["--custom_settings", custom_settings])
        with Popen(command,
            stdout=PIPE,
            bufsize=1,
            universal_newlines=True,
        ) as p:
            for line in p.stdout:
                print(line, end="")

        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)
        
        yield [Path(image) for image in glob.glob(outdir + "/*.png")]

        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
