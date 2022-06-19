# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from subprocess import Popen, PIPE, CalledProcessError
import tempfile
import glob
import models
import majesty


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")        
        models.download_models("/src/models", ongo=True, erlich=True)
        models.download_clip(majesty.clip_load_list)

    def predict(
        self,
        clip_prompt: str = Input(description="Prompt for CLIP guidance"),
        latent_prompt: str = Input(description="Prompt for latent diffusion"),
        model: str = Input(description="Latent diffusion model", default='finetuned', choices=["original", "finetuned", "ongo", "erlich"]),
        settings: Path = Input(description="Advanced settings file", default=None),
    ) -> Path:
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
                "-m",
                "/src/models",
                "-o",
                outdir,
                "--model_source",
                "https://models.nmb.ai/majesty",
            ]
        if settings:
            command.append(["c", settings])            
        with Popen(command,
            stdout=PIPE,
            bufsize=1,
            universal_newlines=True,
        ) as p:
            for line in p.stdout:
                print(line, end="")

        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)

        images = glob.glob(outdir + "/*.png")
        return Path(images[0])

        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
