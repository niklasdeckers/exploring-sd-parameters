import logging
import time
import numpy as np
import torch
from PIL import Image as PILImage
from PIL.Image import Image
from abc import abstractmethod, ABC
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, AutoencoderKL, LCMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from functools import partial
from nicegui import binding
from torch import Tensor
from transformers import CLIPImageProcessor



class GeneratorBase(ABC):

    def __init__(self):
        self.latest_images = []

    @abstractmethod
    def generate_image(self, embedding: Tensor | tuple[Tensor, Tensor]) -> list[Image]:
        pass

    def get_latest_images(self) -> list[Image]:
        """
        Returns the latest generated images in the "cache" and clears the cache.
        This is useful to remove already displayed images from the memory.
        """
        latest_images = self.latest_images
        self.latest_images = []
        return latest_images

    def clear_latest_images(self) -> None:
        self.latest_images = []


class Generator(GeneratorBase):
    height = binding.BindableProperty()
    width = binding.BindableProperty()
    batch_size = binding.BindableProperty()
    num_inference_steps = binding.BindableProperty()
    guidance_scale = binding.BindableProperty()
    use_negative_prompt = binding.BindableProperty()

    @torch.no_grad()
    def __init__(self,
                 batch_size: int = None,
                 hf_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 lora_name: str = "latent-consistency/lcm-lora-sdxl",
                 cache_dir: str | None = '/cache/',
                 num_inference_steps: int = 20,
                 device: str = 'cuda',
                 guidance_scale: float = 7.,
                 use_negative_prompt: bool = False,
                 callback=None,
                 pipe=None,
                 initial_latent_seed: int = 42,
                 # todo this is unused here, but currently passed from config.yaml. should be removed
                 height: int = 1024,
                 width: int = 1024,
                 ):
        """
        Setting the image generation scheduler, SD pipeline, and latents that stay constant during the iterative refining.

        Args:
            hf_model_name: Huggingface model identifier, default is Stable Diffusion XL
            cache_dir: directory to download to model to
            num_inference_steps: number of denoising steps for the model to take
            batch_size: number of images that should be generated in a batch, lower means less vram needed
            device: gpu or cpu that should be used to generate images
            height: image height in pixels. SDXL MIGRATION: was hardcoded to 512 (SD1.5 native res);
                now configurable since SDXL's native resolution is 1024.
            width: image width in pixels. See `height`.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_negative_prompt = use_negative_prompt
        self.callback = callback

        self.device = torch.device("cuda") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        self.initial_latent_generator = torch.Generator(device=self.device)
        self.initial_latent_seed = initial_latent_seed
        self.initial_latent_generator.manual_seed(self.initial_latent_seed)

        # SDXL MIGRATION: StableDiffusionXLPipeline has no safety_checker component at all (unlike
        # StableDiffusionPipeline), so the safety_checker/requires_safety_checker kwargs that used
        # to be passed here have been dropped rather than repointed.
        self.pipe = pipe if pipe else StableDiffusionXLPipeline.from_pretrained(
            hf_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        # SDXL MIGRATION: swapped to the SDXL-specific LCM-LoRA. The SD1.5 LoRA
        # ("latent-consistency/lcm-lora-sdv1-5") targets the single-encoder/768-dim UNet
        # cross-attention layers and is not compatible with SDXL's UNet.
        self.pipe.load_lora_weights(lora_name)
        # QUANTIZATION: fusing a LoRA adapter into bitsandbytes-quantized (int8) weights is
        # unsupported/error-prone (diffusers issue #10550, #10492) since the merge needs floating
        # point weights. A quantized UNet keeps the LoRA adapter unfused instead — PEFT still
        # applies it correctly at inference, just as a separate low-rank pass alongside the frozen
        # quantized base weights.
        if not getattr(self.pipe.unet, "is_quantized", False):
            pipe.fuse_lora()

        # self.pipe.unet = torch.compile(self.pipe.unet, backend="cudagraphs")

        # self.pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=self.pipe.device, dtype=self.pipe.dtype)
        # self.pipe.vae = torch.compile(self.pipe.vae, backend="cudagraphs")

        self.latent_height = int(self.height // self.pipe.vae_scale_factor)
        self.latent_width = int(self.width // self.pipe.vae_scale_factor)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("Cannot use xformers memory efficient attention (maybe xformers not installed)")

        # SAFETY CHECKER: StableDiffusionXLPipeline has no built-in safety_checker component (see
        # note above), so NSFW filtering is applied explicitly here as a post-generation step,
        # reusing the same CLIP-based checker the SD1.5 pipeline used to run internally.
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(device=self.device)
        self.safety_feature_extractor = CLIPImageProcessor.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        )

        self.negative_prompt_embeds = None
        self.negative_pooled_prompt_embed = None
        self.negative_prompt = ""
        if self.use_negative_prompt:
            self.negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, illustration, painting, drawing, art, sketch,bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, worst quality, cropped, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, deformed, missing limb, bad hands, extra digits, extra fingers, not enough fingers, floating head, disembodied"
            # SDXL MIGRATION: the old code tokenized once and called `self.pipe.text_encoder(...)`
            # directly, which assumed a single tokenizer/text_encoder pair. SDXL pipelines expose
            # `tokenizer`/`tokenizer_2` and `text_encoder`/`text_encoder_2`, and additionally require
            # a *pooled* embedding (from text_encoder_2's pooler output) for `added_cond_kwargs`.
            # Since this is just a static negative string (not part of the axis-interpolation math
            # in UserProfileHost), we use the pipeline's own `encode_prompt` helper to get correctly
            # concatenated dual-encoder embeddings and the pooled embedding in one call, rather than
            # hand-rolling the dual tokenizer/encoder calls.
            _, self.negative_prompt_embed, _, self.negative_pooled_prompt_embed = self.pipe.encode_prompt(
                prompt="",
                device=self.pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=self.negative_prompt,
            )

    @torch.no_grad()
    def run_safety_checker(self, images: list[Image]) -> list[Image]:
        """
        Replaces any image flagged as NSFW by the CLIP-based safety checker with a blacked-out
        image, keeping the list the same length/order so it still lines up 1:1 with the
        embeddings that produced it (the recommender relies on that alignment for scoring).
        """
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt").to(self.device)
        images_np = np.stack([np.array(image).astype(np.float32) / 255.0 for image in images])
        checked_images_np, has_nsfw_concept = self.safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values,
        )
        if any(has_nsfw_concept):
            logging.warning(f"Safety checker flagged {sum(has_nsfw_concept)} image(s); blacked out.")
        return [PILImage.fromarray((image * 255).round().astype("uint8")) for image in checked_images_np]

    @torch.no_grad()
    def generate_image(self, embeddings: Tensor, pooled_embeddings: Tensor, latents: Tensor, loading_progress,
                       queue_lock) -> list[Image]:
        """
        Generates a list of image(s) from given embedding

        Args:
        embeddings (Tensor):
            A batch of embeddings as tensor of shape (batch, 77, 2048).
        pooled_embeddings (Tensor):
            SDXL MIGRATION: new required argument. A batch of pooled embeddings of shape
            (batch, 1280), fed into `added_cond_kwargs` via `pooled_prompt_embeds`. SDXL's pipeline
            raises an error if `prompt_embeds` is passed without a matching `pooled_prompt_embeds`.
            See UserProfileHost.inv_transform for how these are derived.
        Returns:
            `list[PIL.Image.Image]: a list of batch many PIL images generated from the embeddings.
        """
        if embeddings.dtype != self.pipe.dtype:
            embeddings = embeddings.type(self.pipe.dtype)
        embeddings = embeddings.to(self.pipe.device)
        pooled_embeddings = pooled_embeddings.to(device=self.pipe.device, dtype=self.pipe.dtype)
        latents = latents.to(self.pipe.device)
        latents = latents.type(self.pipe.dtype)

        pos_prompt_embeds = embeddings
        num_embeddings = pos_prompt_embeds.shape[0]
        batch_steps = self.batch_size or num_embeddings

        images = []
        for i in range(0, num_embeddings, batch_steps):
            task = lambda: self.pipe(height=self.height,
                                     width=self.width,
                                     num_images_per_prompt=1,
                                     prompt_embeds=pos_prompt_embeds[i:i + batch_steps],
                                     pooled_prompt_embeds=pooled_embeddings[i:i + batch_steps],
                                     negative_prompt_embeds=self.negative_prompt_embed.repeat(batch_steps, 1,
                                                                                              1) if self.use_negative_prompt else None,
                                     negative_pooled_prompt_embeds=self.negative_pooled_prompt_embed.repeat(
                                         batch_steps, 1) if self.use_negative_prompt else None,
                                     num_inference_steps=self.num_inference_steps,
                                     guidance_scale=self.guidance_scale,
                                     latents=latents[i:i + batch_steps],
                                     # SDXL MIGRATION: original_size/crops_coords_top_left/target_size
                                     # (SDXL's other new conditioning inputs) are left at the
                                     # pipeline's defaults (derived from height/width above). Expose
                                     # them as explicit args here if non-square/cropped conditioning
                                     # is ever needed.
                                     callback_on_step_end=partial(self.callback,
                                                                  current_step=i,
                                                                  num_embeddings=num_embeddings,
                                                                  loading_progress=loading_progress,
                                                                  batch_size=batch_steps,
                                                                  num_steps=self.num_inference_steps
                                                                  )
                                     ).images

            result = queue_lock.do_work(task)
            images.extend(result.result())
        images = self.run_safety_checker(images)
        self.latest_images.extend(images)
        return images



