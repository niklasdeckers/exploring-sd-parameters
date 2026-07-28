import torch
from diffusers import StableDiffusionXLPipeline
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
        pipe = StableDiffusionXLPipeline.from_pretrained(
            global_args.hf_model_name,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        # SDXL MIGRATION: the larger SDXL UNet takes noticeably more VRAM and compile time under
        # torch.compile than SD1.5's; re-check memory headroom (especially with batch_size=10).
        pipe.unet = torch.compile(pipe.unet, backend="cudagraphs")
        # SDXL MIGRATION: the SD1.5-specific "stabilityai/sd-vae-ft-mse" VAE swap was dropped — it
        # is a checkpoint fine-tuned for (and only numerically valid with) SD1.5's latent space.
        # We now just keep + compile the VAE bundled with the SDXL pipeline checkpoint instead of
        # swapping in a separate one. If VAE instability shows up under bfloat16 (a known issue
        # with the stock SDXL VAE), consider pointing at "madebyollin/sdxl-vae-fp16-fix" here.
        pipe.vae = torch.compile(pipe.vae, backend="cudagraphs")

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
                 reconnect_timeout=global_args.reconnect_timeout)
        start()
