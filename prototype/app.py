import torch
from diffusers import BitsAndBytesConfig, StableDiffusionXLPipeline, UNet2DConditionModel
from nicegui import ui as ngUI

from prototype.utils import ProducerConsumer
from prototype.webuserinterface import WebUI
from prototype.generator.generator import Generator

global_args = None
pipe = None
queue_lock = ProducerConsumer()  # QLock()


@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    global pipe
    global generator
    ui = await WebUI.create(global_args, pipe, generator, queue_lock)
    ui.run()


@ngUI.page('/')
def start():
    """
    Just redirects to '/demo', because '/' is the auto-index page.
    """
    ngUI.navigate.to('/demo')


class App:
    """
    The entry point into the application.
    """

    def __init__(self, args):
        global global_args
        global_args = args
        self.device = torch.device("cuda") if (
                global_args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        # Initialize a central StableDiffusionXLPipeline for all sessions
        # SDXL MIGRATION: StableDiffusionXLPipeline has no safety_checker component, so the
        # safety_checker/requires_safety_checker kwargs used for the SD1.5 pipeline were dropped.
        global pipe
        # QUANTIZATION: the UNet is the only component worth quantizing here (VAE/text encoders
        # are already small per diffusers' own bitsandbytes guidance). Loaded separately so it can
        # carry its own quantization_config, then handed into the pipeline via the `unet=` kwarg.
        # `quantize_unet` is read from configs/config.yaml (null/"8bit"/"4bit").
        pipe_kwargs = {}
        if global_args.quantize_unet == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif global_args.quantize_unet == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif global_args.quantize_unet is None:
            quantization_config = None
        else:
            raise ValueError(f"Unknown quantize_unet config value: {global_args.quantize_unet!r}")

        if quantization_config is not None:
            pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                global_args.hf_model_name,
                subfolder="unet",
                cache_dir=global_args.path.cache_dir,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            global_args.hf_model_name,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
            **pipe_kwargs,
        ).to(device=self.device)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        # SDXL MIGRATION: the larger SDXL UNet takes noticeably more VRAM and compile time under
        # torch.compile than SD1.5's; re-check memory headroom (especially with batch_size=10).
        # pipe.unet = torch.compile(pipe.unet, backend="cudagraphs")
        # SDXL MIGRATION: the SD1.5-specific "stabilityai/sd-vae-ft-mse" VAE swap was dropped — it
        # is a checkpoint fine-tuned for (and only numerically valid with) SD1.5's latent space.
        # We now just keep + compile the VAE bundled with the SDXL pipeline checkpoint instead of
        # swapping in a separate one. If VAE instability shows up under bfloat16 (a known issue
        # with the stock SDXL VAE), consider pointing at "madebyollin/sdxl-vae-fp16-fix" here.
        # pipe.vae = torch.compile(pipe.vae, backend="cudagraphs")

        global generator
        generator = Generator(
                cache_dir=args.path.cache_dir,
                device=args.device,
                hf_model_name=args.hf_model_name,
                pipe=pipe,
                **args.generator,
            )

    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port,
                 reconnect_timeout=global_args.reconnect_timeout, reload=False)
        start()
