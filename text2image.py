import argparse
from PIL import Image

from io import BytesIO
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL

import litserve as ls


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="OnomaAIResearch/Illustrious-xl-early-release-v0",
    )
    parser.add_argument("--vae", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: str = "bad quality, worst quality, lowres, bad anatomy, sketch, jpeg artifacts, ugly, poorly drawn, signature, watermark, bad anatomy, bad hands, bad feet, retro, old, 2000s, 2010s, 2011s, 2012s, 2013s, multiple views, screencap"
    inference_steps: int = 25
    cfg_scale: float = 6.5
    width: int = 768
    height: int = 1024

    @field_validator("width", "height")
    def check_divisible_by_64(cls, value):
        if value % 64 != 0:
            raise ValueError(f"{value} is not divisible by 64")
        return value


class T2IModel:
    def __init__(self, model_name: str, vae_name: str) -> None:
        vae = AutoencoderKL.from_pretrained(
            vae_name,
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_name,
            vae=vae,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            add_watermarker=False,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        self.pipe = pipe

    def generate(
        self,
        params: GenerationParams,
    ):
        image = self.pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.inference_steps,
            guidance_scale=params.cfg_scale,
            width=params.width,
            height=params.height,
            return_type="pil",
        ).images[0]  # type: ignore

        return image


class SimpleLitAPI(ls.LitAPI):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.vae_name = args.vae

    def setup(self, device):
        self.model = T2IModel(self.model_name, self.vae_name)
        self.model.pipe.to(device)

    def decode_request(self, request: dict):
        params = GenerationParams(**request)
        return params

    def predict(self, params: GenerationParams):
        image = self.model.generate(params)
        return image

    def encode_response(self, image: Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="WEBP")

        return Response(
            content=buffered.getvalue(), headers={"Content-Type": "image/webp"}
        )


def main():
    args = prepare_args()
    server = ls.LitServer(SimpleLitAPI(args), accelerator="auto", max_batch_size=1)
    server.run(port=args.port)


if __name__ == "__main__":
    main()
