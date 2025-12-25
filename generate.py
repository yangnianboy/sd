import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.animate import animate_two_view
import torchvision.transforms.functional as TF

from config import *
from utils import flush, im_to_np

class VisualAnagramGenerator:
    def __init__(self, hf_token):
        from huggingface_hub import login
        login(token=hf_token)
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.prompts = None
        self.views = None
        self.stage_1 = None
        self.stage_2 = None
        self.stage_3 = None

    def encode_prompts(self, prompts):
        self.prompts = prompts
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_STAGE_1,
            subfolder="text_encoder",
            device_map="auto",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_STAGE_1,
            text_encoder=text_encoder,
            unet=None
        )
        pipe = pipe.to(DEVICE)
        
        prompt_embeds = [pipe.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        self.prompt_embeds = torch.cat(prompt_embeds)
        self.negative_prompt_embeds = torch.cat(negative_prompt_embeds)
        
        del text_encoder
        del pipe
        flush()
        flush()

    def load_models(self):
        self.stage_1 = DiffusionPipeline.from_pretrained(
            MODEL_STAGE_1,
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.stage_1.enable_model_cpu_offload()
        self.stage_1.to(DEVICE)

        self.stage_2 = DiffusionPipeline.from_pretrained(
            MODEL_STAGE_2,
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.stage_2.enable_model_cpu_offload()
        self.stage_2.to(DEVICE)

        self.stage_3 = DiffusionPipeline.from_pretrained(
            MODEL_STAGE_3,
            torch_dtype=torch.float16
        )
        self.stage_3.enable_model_cpu_offload()
        self.stage_3 = self.stage_3.to(DEVICE)

    def set_views(self, view_types):
        self.views = get_views(view_types)

    def generate_64(self):
        return sample_stage_1(
            self.stage_1,
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.views,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            reduction='mean',
            generator=None
        )

    def generate_256(self, image_64):
        return sample_stage_2(
            self.stage_2,
            image_64,
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.views,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            reduction='mean',
            noise_level=NOISE_LEVEL,
            generator=None
        )

    def generate_1024(self, image_256):
        image_1024 = self.stage_3(
            prompt=self.prompts[0],
            image=image_256,
            noise_level=0,
            output_type='pt',
            generator=None
        ).images
        return image_1024 * 2 - 1

    def generate(self, prompts, view_types=None):
        if view_types is None:
            view_types = DEFAULT_VIEW
        
        self.encode_prompts(prompts)
        self.load_models()
        self.set_views(view_types)
        
        image_64 = self.generate_64()
        image_256 = self.generate_256(image_64)
        image_1024 = self.generate_1024(image_256)
        
        return {
            '64': image_64,
            '256': image_256,
            '1024': image_1024
        }

    def save_animation(self, image, save_path='./animation.mp4'):
        im_size = image.shape[-1]
        frame_size = int(im_size * 1.5)
        pil_image = TF.to_pil_image(image[0] / 2. + 0.5)
        
        animate_two_view(
            pil_image,
            self.views[1],
            self.prompts[0],
            self.prompts[1],
            save_video_path=save_path,
            hold_duration=120,
            text_fade_duration=10,
            transition_duration=45,
            im_size=im_size,
            frame_size=frame_size,
        )
        return save_path

    def get_view_images(self, image):
        return [im_to_np(view.view(image[0])) for view in self.views]

