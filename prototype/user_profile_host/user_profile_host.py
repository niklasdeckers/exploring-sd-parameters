import json
import random
import torch
from diffusers import StableDiffusionXLPipeline
from functools import partial
from nicegui import binding
from sklearn.decomposition import PCA
from torch import Tensor

from .optimizer import *
from .recommender import *
from ..constants import RecommendationType


def build_axis_prompts(axis_style: str, n_embedding_axis: int, original_prompt: str, image_styles: list,
                        secondary_contexts: list, atmospheric_attributes: list, quality_terms: list,
                        use_embedding_center: bool, rng: random.Random) -> list:
    """
    Builds the n_embedding_axis axis-prompts used to fit the hyperspherical basis for the user
    profile space (see UserProfileHost.load_user_profile_host). Every axis_style must return
    exactly n_embedding_axis prompts — a mismatch here silently propagates into a confusing
    `mat1 and mat2 shapes cannot be multiplied` error deep inside inv_transform() instead, so the
    fixed-template styles ('random'/'simple'/'complex') are checked explicitly below.
    """
    if axis_style == 'ordered':
        return [
            rng.choice(image_styles) + original_prompt + rng.choice(secondary_contexts) +
            rng.choice(atmospheric_attributes) + rng.choice(quality_terms)
            for _ in range(n_embedding_axis)
        ]
    elif axis_style == 'random':
        templates = [
            "A beautiful purple flower in a dark forest, in the style of hyper-realistic sculptures, with dark orange and green colors, set against post-apocalyptic backdrops with light red and yellow hues. it is displayed in museum gallery dioramas, featuring soft, dreamy scenes with an orange and green surreal 8k zbrush render.",
            "Fluid abstract background, dark indigo, art, behance",
            "hyperdetailed eyes, tee-shirt design, line art, black background, ultra detailed artistic, detailed gorgeous face, natural skin, water splash, colour splash art, fire and ice, splatter, black ink, liquid melting, dreamy, glowing, glamour, glimmer, shadows, oil on canvas, brush strokes, smooth, ultra high definition, 8k, unreal engine 5, ultra sharp focus, intricate artwork masterpiece, ominous, golden ratio, highly detailed, vibrant, production cinematic character render, ultra high quality model",
            "Futuristic sci-fi pod chair, flat design, product-view, editorial photography",
            "Cute girl behind window, rainy, photography surreal art, blurry, minimalistic",
            "Shadowy figure of a woman emerging from the darkness, black and grey gradient, foggy, realistic, 8k resolution, unreal engine, cinematic",
            "Old man standing next to a giant monster, in the style of contemporary vintage photography, necronomicon illustrations, tabletop photography, 1890, hyperrealistic animal portraits, ghostly presence, whirring contrivances",
            "Victo ngai style",
            "Detailed, vibrant illustration of a cowboy in the copper canyons the sierra of chihuahua state, by herge, in the style of tin-tin comics, vibrant colors, detailed, sunny day, attention to detail, 8k",
            "Create a surreal desert with alien plants, the plants are shaped like canary_yellow_perlwhite, are partially transparent with tentacles and spines, in the sand laying pearls, backdrop is the storm of cosmic dust and cosmic clouds the heaven is dark colored unreal engine 6 color palette knives painting oel on canvas conzeptart, high qualty, cinema_stil, wide shot",
            "beautiful field of flowers, colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
            "A boy playing video games at night in his room, illustration by hergé, perfect coloring, 8k",
            "Drawing of a cosmic extraterrestrial technology healing chamber, with many cables connecting the chamber to a large translucent transparent crystal. a body silhouette inside. ambient aircraft.",
            "An intricate village made of psychedelic mushrooms, art by greg rutkowsk, 3d render",
            "Some people look over tall building windows, in the style of dark hues, rural china, coded patterns, sparse and simple, uhd image, urbancore, sovietwave, negative space, award-winning design",
            "Diesel-punk hip-hop punk ashigaru wearing diesel-punk oni armor. full body fighting pose. traditional wet ink and watercolor painting style. black, grey, red, and metallic gold ink. gestural speed paint by artgerm and jungshan. street fighter style.",
            "A cute minimalistic simple capybara side profile, in the style of jon klassen, desaturated light and airy pastel color palette, nursery art, white background",
            "black and red ink, a crane in chinese style, ink art by mschiffer, whimsical, rough sketch, (sketch1.3)",
            "A cute cartoon girl in a dress holding a white kitten, full body, yellow background, keith haring style doodle, sharpie illustration, bold lines and solid colors, simple details, (((minimalism))), yellow background",
            "Japanese animation, panoramic, colorful, a small corgi with closed eyes backstroke in the pool, most of the picture shows water, corgi accounts for a small part of the picture, water is light blue transparent and clear, water ripple texture is clear, light refraction, corgi and water are not fuzzy, in hd, phone wallpaper size, hd, 32k",
            "Body portrait photography, in a smoke-filled office full of cables and wires and led, featuring a carbon motor head, an attractive transparent white plexiglass secretary robot reading an ancient book at her desk, 80-degree view. art by sergio lopez, natalie shau, james jean, and salvador dali."
        ]
        _assert_enough_templates(axis_style, templates, n_embedding_axis)
        add_ons = templates[:n_embedding_axis]
        # Include original prompt if not using the embedding center to remain the primary context
        if not use_embedding_center:
            add_ons = [original_prompt + ', ' + a for a in add_ons]
        return add_ons
    elif axis_style == 'simple':
        templates = [
            "in the style of a surreal oil painting",
            "as a vintage photograph from the 1920s",
            "drawn like a Studio Ghibli animation",
            "rendered as hyper-realistic 3D CGI",
            "in minimalist flat vector art style",
            "as a charcoal sketch on parchment",
            "in the aesthetic of vaporwave",
            "painted in watercolor with soft pastel tones",
            "illustrated like a medieval manuscript",
            "as a pixel art scene from an 8-bit video game",
            "set in a dense futuristic megacity",
            "inside a sunlit forest clearing",
            "floating above the clouds at golden hour",
            "underwater in a bioluminescent reef",
            "in a vast desert with ancient ruins",
            "on a snowy mountain peak during a blizzard",
            "on a distant alien planet with purple skies",
            "in a neon-lit alleyway at midnight",
            "at the bottom of a dark cave",
            "inside a massive ancient library",
            "during the last moments of a sunset",
            "in a post-apocalyptic future",
            "on a quiet early morning",
            "in the distant future, year 4000",
            "during a Renaissance-era festival",
            "with an eerie, unsettling atmosphere",
            "filled with joyous, playful energy",
            "with a dreamlike, ethereal mood",
            "with dark and mysterious undertones",
            "evoking nostalgia and melancholy",
            "bursting with chaotic and surreal energy",
            "calm, serene, and meditative",
            "exploring the theme of isolation",
            "representing the passage of time",
        ]
        _assert_enough_templates(axis_style, templates, n_embedding_axis)
        add_ons = templates[:n_embedding_axis]
        # Include original prompt if not using the embedding center to remain the primary context
        if not use_embedding_center:
            add_ons = [original_prompt + ', ' + a for a in add_ons]
        return add_ons
    elif axis_style == 'complex':
        templates = [
            f"A beautiful purple flower in a dark forest, depicting {original_prompt}, in the style of hyper-realistic sculptures, with dark orange and green colors, set against post-apocalyptic backdrops with light red and yellow hues. It is displayed in museum gallery dioramas, featuring soft, dreamy scenes with an orange and green surreal 8k ZBrush render.",
            f"Fluid abstract background featuring {original_prompt}, dark indigo palette, artistic textures, Behance-style presentation.",
            f"Hyperdetailed artistic portrait of {original_prompt}, tee-shirt design, line art on black background, ultra detailed face, natural skin, water and colour splashes, fire and ice contrast, oil on canvas, brush strokes, vibrant and cinematic render with the golden ratio in ultra high definition, 8k Unreal Engine 5 quality.",
            f"A futuristic sci-fi pod chair designed to accommodate {original_prompt}, flat design, product-view layout, editorial photography style.",
            f"Cute girl behind window thinking about {original_prompt}, rainy scene, photography surreal art, blurry and minimalistic tones.",
            f"Shadowy figure representing {original_prompt} emerging from the darkness, in a foggy and cinematic black and grey gradient, 8k resolution Unreal Engine render.",
            f"An old man standing next to a giant monster, as a metaphor for {original_prompt}, in the style of contemporary vintage photography, Necronomicon illustrations, 1890 tabletop hyperrealistic animal portraiture.",
            f"Victo Ngai’s visual interpretation of {original_prompt}, blending magical realism and intricate detailing.",
            f"Detailed, vibrant illustration of a cowboy experiencing {original_prompt} in the copper canyons of Chihuahua, drawn in the style of Tintin comics by Hergé, 8k sunny detailed coloring.",
            f"Create a surreal desert with alien plants embodying {original_prompt}, canary yellow and pearl white, partially transparent with tentacles, storm of cosmic dust in the background, painted with Unreal Engine 6 palette knives in cinematic concept art style.",
            f"Beautiful field of flowers surrounding {original_prompt}, colorful bloom everywhere with perfect lighting, Leica Summicron 35mm f2.0, Kodak Portra 400, film grain aesthetics.",
            f"A boy playing video games late at night, imagining {original_prompt}, drawn by Hergé in 8k coloring and comic style.",
            f"A cosmic extraterrestrial healing chamber, inside which lies {original_prompt}, with translucent crystals and connecting ambient aircraft cables, ambient concept render.",
            f"An intricate village made of psychedelic mushrooms where {original_prompt} lives, in the art style of Greg Rutkowski, 3D rendered.",
            f"From a high-rise window, people witness {original_prompt} in rural China, designed with dark hues, sparse negative space, coded patterns, Sovietwave and urbancore elements, UHD.",
            f"Diesel-punk hip-hop samurai version of {original_prompt}, wearing oni armor, posed mid-fight, painted in traditional wet ink and watercolor by Artgerm and Jungshan, with metallic gold and gestural brush strokes.",
            f"A cute minimalistic capybara, side profile, accompanied by {original_prompt}, drawn in the pastel nursery art style of Jon Klassen with desaturated airy tones.",
            f"A majestic crane soaring beside {original_prompt}, rendered in whimsical Chinese-style black and red ink by MSchiffer, rough sketch style.",
            f"A cute cartoon girl in a dress holding a white kitten and a drawing of {original_prompt}, on a yellow background, Keith Haring doodle style, bold lines and solid colors.",
            f"Japanese animation panoramic scene: a small corgi backstroking in a pool while {original_prompt} floats nearby, light blue water with ripples and transparent refraction, 32k HD phone wallpaper aesthetic.",
            f"A body portrait photography scene inside a smoky LED-lit office, {original_prompt} portrayed as a transparent white plexiglass robot secretary reading an ancient book, viewed at 80 degrees. Art by Sergio Lopez, Natalie Shau, James Jean, Salvador Dalí."
        ]
        _assert_enough_templates(axis_style, templates, n_embedding_axis)
        return templates[:n_embedding_axis]
    else:
        raise NotImplementedError()


def _assert_enough_templates(axis_style: str, templates: list, n_embedding_axis: int) -> None:
    if len(templates) < n_embedding_axis:
        raise ValueError(
            f"axis_style={axis_style!r} only has {len(templates)} hardcoded templates, which is "
            f"fewer than n_embedding_axis={n_embedding_axis}. Lower n_embedding_axis to at most "
            f"{len(templates)}, or choose a different axis_style."
        )


class UserProfileHost():
    original_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()
    height = binding.BindableProperty()
    width = binding.BindableProperty()
    latent_space_length = binding.BindableProperty()
    n_latent_axis = binding.BindableProperty()
    n_embedding_axis = binding.BindableProperty()
    use_embedding_center = binding.BindableProperty()
    use_latent_center = binding.BindableProperty()
    n_recommendations = binding.BindableProperty()
    ema_alpha = binding.BindableProperty()
    beta = binding.BindableProperty()
    beta_step_size = binding.BindableProperty()
    include_random_rec = binding.BindableProperty()

    # TODO: Group together Recommender Args and just pass them to the recommender, should simplyfy this arg list
    def __init__(
            self,
            original_prompt: str,
            add_ons: list = None,
            recommendation_type: str = RecommendationType.RANDOM,
            stable_dif_pipe: StableDiffusionXLPipeline = None,
            hf_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
            cache_dir: str = './cache/',
            n_embedding_axis: int = 13,
            use_embedding_center: bool = True,
            n_latent_axis: int = 3,
            use_latent_center: bool = False,
            n_recommendations: int = 6,
            include_random_recommendations: bool = False,
            ema_alpha: float = 0.5,
            beta: float = 0.3,
            beta_step_size: float = 0.1,
            latent_axes_seed: int = 42,
            recommendation_seed: int = 42,
            initial_recommendation_seed: int = 43,
            prompts_seed: int = 42,
            axis_style: str = 'ordered',
            latent_space_length: float = 15.00,
            original_prompt_share=0.0,
            # SDXL MIGRATION: was hardcoded to 512 (SD1.5 native res); now configurable since SDXL's
            # native resolution is 1024.
            height: int = 1024,
            width: int = 1024,
    ):
        """
        This class is the main interface for the user profile host. It initializes the user profile host with the
        :param original_prompt: The original prompt as string.
        :param add_ons: A list of additional prompts to be used as axis for the user profile space.
            Elements of the list are strings.
        :param recommendation_type: The type of recommender to be used. Must be in constants.RecommendationType
        :param stable_dif_pipe: If given, the pipeline will be used to calculate the CLIP embeddings.
        Otherwise, a new pipeline will be created.
        :param hf_model_name: Name of the Hugging Face model.
        :param cache_dir: Path to the cache directory.
        :param n_embedding_axis: Number of axis to be used for the user profile.
        :param use_embedding_center: Whether to use the original prompt as the center of the user profile space.
        :param n_latent_axis: Number of latent axis to be used for the user profile.
        :param use_latent_center: Whether to use a latent center instead of all zeros.
        :param n_recommendations: Number of recommendations to be generated each iteration.
        :param ema_alpha: Used for an exponential moving average to update the user profile.
            Factor for the exponential moving average. Higher values give more weight to recent recommendations.
        :param beta: Trade-off between exploration and exploitation. Must be in [0, 1]. 0 means exploration, 1 means
            exploitation. Beta is increased after each recommendation (i.e. more exploitation).
        :param beta_step_size: The step size for the beta increase.
        """
        # Some Clip Hyperparameters
        self.original_prompt = original_prompt
        self.add_ons = add_ons
        self.recommendation_type = recommendation_type
        self.stable_dif_pipe = stable_dif_pipe
        # SDXL MIGRATION: SD1.5's single CLIP-L encoder produced 768-dim per-token embeddings.
        # SDXL concatenates CLIP-L (768-dim) and OpenCLIP-bigG (1280-dim) per-token hidden states,
        # for a combined 2048-dim embedding (see clip_embedding()).
        self.embedding_dim = 2048
        self.n_clip_tokens = 77
        self.height = height
        self.width = width
        self.latent_space_length = latent_space_length
        self.n_latent_axis = (
                    n_latent_axis * 2) if self.recommendation_type == RecommendationType.SIMPLE else n_latent_axis
        self.n_embedding_axis = n_embedding_axis
        self.use_embedding_center = use_embedding_center
        self.use_latent_center = use_latent_center
        self.n_recommendations = n_recommendations
        self.ema_alpha = ema_alpha
        self.beta = min(beta, 1.)
        self.beta_step_size = beta_step_size
        self.include_random_rec = include_random_recommendations
        self.axis_style = axis_style
        self.latent_axes_seed = latent_axes_seed
        self.recommendation_seed = recommendation_seed
        self.initial_recommendation_seed = initial_recommendation_seed
        self.prompts_seed = prompts_seed  # seed for random prompt selection
        self.original_prompt_share = original_prompt_share

        # intelligent prompt generation
        if self.recommendation_type == RecommendationType.SIMPLE:
            self.recommendation_prompt_generator = random.Random(self.recommendation_seed)

        # Check for valid values
        assert self.beta >= 0., "Beta should be in range [0., 1.]"
        assert self.beta_step_size >= 0. and self.beta_step_size < 1., "Beta Step Size should be in [0., 1.]"

        # Placeholder for the already evaluated embeddings of the current user
        self.embeddings = None
        self.preferences = torch.tensor([])

        # Placeholder until the user_profile is fit the first time
        self.user_profile = None

        # Holds previous low dimensional user profiles
        self.user_profile_history = []

        # Bounds remain fixed to 0., 1. for simplicity
        self.embedding_bounds = [0., 1.]
        self.latent_bounds = [0., 1.]

        # Initialize tokenizer and text encoder to calculate CLIP embeddings
        if not self.stable_dif_pipe:
            self.stable_dif_pipe = StableDiffusionXLPipeline.from_pretrained(
                pretrained_model_name_or_path=hf_model_name,
                cache_dir=cache_dir
            )
        # SDXL MIGRATION: SDXL pipelines expose two tokenizer/text_encoder pairs (CLIP-L +
        # OpenCLIP-bigG) instead of SD1.5's single pair. `self.tokenizer`/`self.text_encoder` keep
        # pointing at the first (CLIP-L) pair since existing code below only uses them for
        # `.dtype`/`.device`, which is the same across both encoders in practice.
        self.tokenizer = self.stable_dif_pipe.tokenizer
        self.text_encoder = self.stable_dif_pipe.text_encoder
        self.tokenizer_2 = self.stable_dif_pipe.tokenizer_2
        self.text_encoder_2 = self.stable_dif_pipe.text_encoder_2

        self.load_user_profile_host()

    def load_user_profile_host(self):
        print("Create new profile with prompt:", self.original_prompt)
        # Define the center of the user_space with the original prompt embedding
        # SDXL MIGRATION: clip_embedding() now returns a (sequence_embedding, pooled_embedding)
        # tuple instead of a single tensor; self.pooled_prompt_embedding is the center/fallback
        # pooled vector used throughout inv_transform().
        self.prompt_embedding, self.pooled_prompt_embedding = self.clip_embedding(self.original_prompt)
        self.embedding_length = torch.linalg.vector_norm(self.prompt_embedding, ord=2, dim=-1, keepdim=False)
        if not self.use_embedding_center:
            self.embedding_center = torch.zeros(size=(1, self.n_clip_tokens, self.embedding_dim))
        else:
            self.embedding_center = self.prompt_embedding

        # Generate axis to define the user profile space with extensions of the original user-promt in the clip embedding space
        with open('prototype/user_profile_host/prompt_terms.json', 'r') as f:
            prompt_terms = json.load(f)

        self.image_styles = prompt_terms["image_styles"]
        self.secondary_contexts = prompt_terms["secondary_contexts"]
        self.atmospheric_attributes = prompt_terms["atmospheric_attributes"]
        self.quality_terms = prompt_terms["quality_terms"]

        # Create Add ons with original prompt included at the semantically correct position
        self.generator = random.Random(self.prompts_seed)
        self.add_ons = build_axis_prompts(
            axis_style=self.axis_style,
            n_embedding_axis=self.n_embedding_axis,
            original_prompt=self.original_prompt,
            image_styles=self.image_styles,
            secondary_contexts=self.secondary_contexts,
            atmospheric_attributes=self.atmospheric_attributes,
            quality_terms=self.quality_terms,
            use_embedding_center=self.use_embedding_center,
            rng=self.generator,
        )

        self.embedding_axis = []
        # SDXL MIGRATION: parallel list of pooled embeddings, one per add-on prompt, used below to
        # fit a second hyperspherical basis for pooled-embedding reconstruction.
        self.pooled_embedding_axis = []
        # print('The embedding axis will consist of the following prompts:')
        for prompt in self.add_ons:
            # print(prompt)
            sequence_embedding, pooled_embedding = self.clip_embedding(prompt)
            self.embedding_axis.append(sequence_embedding)
            self.pooled_embedding_axis.append(pooled_embedding)
        self.embedding_axis = torch.stack(self.embedding_axis)
        self.pooled_embedding_axis = torch.stack(self.pooled_embedding_axis)

        # Build user subspace parameters
        if self.recommendation_type in [RecommendationType.HYPERSPHERICAL_RANDOM,
                                        RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                        RecommendationType.HYPERSPHERICAL_BAYESIAN]:
            # SDXL MIGRATION: the naive `[:, -1, :]` indexing picks whatever sits at the final
            # (77th) padded token position, which for prompts shorter than 77 tokens is a
            # pad/EOS-repeat token, not necessarily a meaningful "summary" of the prompt. If you
            # want the primary embedding axis to use the same validated summary-token technique
            # applied to the pooled axis below (`attention_mask.sum(-1) - 2`), this is the place to
            # change it — left as-is here since it wasn't part of the requested SDXL port.
            base_embeddings = self.embedding_axis[:, -1,
                              :].float().cpu().numpy()  # only keep the last token sequence step (which acts as a summary)
            n = base_embeddings.shape[0]  # n_embedding_axis
            k = base_embeddings.shape[-1]  # CLIP dimension (2048 for SDXL: 768 CLIP-L + 1280 OpenCLIP-bigG)

            self.hyperspherical_center, self.hyperspherical_radius, self.hyperspherical_basis = \
                self._fit_circumscribed_hypersphere(base_embeddings)

            # SDXL MIGRATION: a second, parallel hyperspherical basis fit over the pooled-embedding
            # axis (see clip_embedding()), so inv_transform() can reconstruct a pooled embedding for
            # each recommendation using the SAME coefficients as the primary embedding — keeping
            # the two semantically consistent. Only built/used for the HYPERSPHERICAL_* family;
            # other recommender types fall back to a constant (center-prompt) pooled embedding in
            # inv_transform().
            pooled_base_embeddings = self.pooled_embedding_axis.float().cpu().numpy()
            (self.hyperspherical_center_pooled, self.hyperspherical_radius_pooled,
             self.hyperspherical_basis_pooled) = self._fit_circumscribed_hypersphere(pooled_base_embeddings)

            # Our user space only operates on the final token sequence step (out of the 77 tokens), which acts as a
            # summary of the whole token sequence. This means that we have to get back into the (batch x) 77 x 768
            # space to pass back to the image generator. However, the image generator has two typical constraints:
            # The first of the 77 steps is always the same and the other steps should converge (i.e., be identical
            # after some point). We just copy the start token embedding from an existing real embedding to the first
            # position and repeat the recommended (batch x) 1 x 768 embedding to fill all 76 remaining steps.
            def convert_to_full_text(embeddings, k, n_tokens, original_starttoken):
                return torch.cat((original_starttoken.reshape([1, 1, k]).expand(embeddings.shape[0], 1, -1),
                                  # for each resulting full text (out of the batch), get one start token embedding
                                  # (of size CLIP dimension)
                                  embeddings.reshape([-1, 1, k]).expand(-1, n_tokens - 1, -1)),
                                 dim=1)  # expand the single time step to 76 and concat them to the start token embedding.

            self.get_full_text_embeddings = partial(convert_to_full_text, k=k, n_tokens=self.embedding_axis.shape[1],
                                                    original_starttoken=self.embedding_axis[0, 0])

        # Similarly, define axis in the latent space to have variations in both spaces that together build the user space
        if self.n_latent_axis:
            generator = torch.Generator()  # cpu
            generator.manual_seed(self.latent_axes_seed)
            if self.recommendation_type in [RecommendationType.HYPERSPHERICAL_RANDOM,
                                            RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                            RecommendationType.HYPERSPHERICAL_BAYESIAN]:
                # already include the standard deviation here and not via the parameter latent_space_length (will be ignored later)
                self.latent_axis = torch.randn(
                    (self.n_latent_axis, self.stable_dif_pipe.unet.config.in_channels, self.height // 8,
                     self.width // 8), generator=generator) * self.stable_dif_pipe.scheduler.init_noise_sigma
            else:
                self.latent_center = torch.randn((1, self.stable_dif_pipe.unet.config.in_channels, self.height // 8,
                                                  self.width // 8),
                                                 generator=generator) if self.use_latent_center else (
                    torch.zeros(
                        size=(1, self.stable_dif_pipe.unet.config.in_channels, self.height // 8, self.width // 8)))
                self.latent_axis = torch.randn(
                    (self.n_latent_axis, self.stable_dif_pipe.unet.config.in_channels, self.height // 8,
                     self.width // 8), generator=generator)
            self.num_axis = self.embedding_axis.shape[0] + self.latent_axis.shape[0]
        else:
            self.num_axis = self.embedding_axis.shape[0]

        # Generally required
        if self.recommendation_type in [RecommendationType.HYPERSPHERICAL_RANDOM,
                                        RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                        RecommendationType.HYPERSPHERICAL_BAYESIAN]:
            # remove one embedding dimension due to lower-dimensional circumscribed hypersphere
            self.random_recommender = HypersphericalRandomRecommender(n_embedding_axis=self.n_embedding_axis - 1,
                                                                      n_latent_axis=self.n_latent_axis,
                                                                      seed=self.initial_recommendation_seed)
        else:
            self.random_recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                        n_latent_axis=self.n_latent_axis,
                                                        seed=self.initial_recommendation_seed)

        # Initialize Optimizer and Recommender based on one Mode
        if self.recommendation_type == RecommendationType.FUNCTION_BASED:
            self.recommender = BayesianRecommender(n_embedding_axis=self.n_embedding_axis,
                                                   n_latent_axis=self.n_latent_axis, seed=self.recommendation_seed)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.RANDOM:
            self.recommender = RandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                 n_latent_axis=self.n_latent_axis, seed=self.recommendation_seed)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.EMA_DIRICHLET:
            self.recommender = DirichletRecommender(n_embedding_axis=self.n_embedding_axis,
                                                    n_latent_axis=self.n_latent_axis, seed=self.recommendation_seed)
            self.optimizer = EMAWeightedSumOptimizer(n_recommendations=self.n_recommendations, alpha=self.ema_alpha)
        elif self.recommendation_type == RecommendationType.BASELINE:
            self.recommender = BaselineRecommender(n_latent_axis=self.n_latent_axis,
                                                   in_channels=self.stable_dif_pipe.unet.config.in_channels,
                                                   height=self.height, width=self.width,
                                                   init_noise_sigma=self.stable_dif_pipe.scheduler.init_noise_sigma,
                                                   seed=self.recommendation_seed)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.SIMPLE:
            self.recommender = SimpleRandomRecommender(n_embedding_axis=self.n_embedding_axis,
                                                       n_latent_axis=self.n_latent_axis)
            self.optimizer = SimpleOptimizer(n_embedding_axis=self.n_embedding_axis,
                                             n_latent_axis=self.n_latent_axis,
                                             image_styles=self.image_styles,
                                             secondary_contexts=self.secondary_contexts,
                                             atmospheric_attributes=self.atmospheric_attributes,
                                             quality_terms=self.quality_terms)
        elif self.recommendation_type == RecommendationType.HYPERSPHERICAL_RANDOM:
            self.recommender = HypersphericalRandomRecommender(n_embedding_axis=self.n_embedding_axis - 1,
                                                               n_latent_axis=self.n_latent_axis,
                                                               seed=self.recommendation_seed)
            self.optimizer = NoOptimizer()
        elif self.recommendation_type == RecommendationType.HYPERSPHERICAL_MOVING_CENTER:
            self.recommender = HypersphericalMovingCenterRecommender(n_embedding_axis=self.n_embedding_axis - 1,
                                                                     n_latent_axis=self.n_latent_axis,
                                                                     seed=self.recommendation_seed)
            self.optimizer = HypersphericalEMAOptimizer(n_recommendations=self.n_recommendations,
                                                        n_embedding_axis=self.n_embedding_axis - 1,
                                                        n_latent_axis=self.n_latent_axis, alpha=self.ema_alpha)
        elif self.recommendation_type == RecommendationType.HYPERSPHERICAL_BAYESIAN:
            self.recommender = HypersphericalBayesianRecommender(n_embedding_axis=self.n_embedding_axis - 1,
                                                                 n_latent_axis=self.n_latent_axis,
                                                                 seed=self.recommendation_seed)
            self.optimizer = NoOptimizer()
        else:
            raise ValueError(f"The recommendation type {self.recommendation_type} is not implemented yet.")

    @staticmethod
    def _fit_circumscribed_hypersphere(base_embeddings):
        """
        Fits the circumscribed hypersphere through `base_embeddings` (n_points, dim): finds the
        center lying on their hyperplane and equidistant from all of them, plus an orthonormal
        basis of the (dim-1)-dimensional hyperplane through that center.

        SDXL MIGRATION: factored out of `load_user_profile_host` (the math itself is unchanged
        from the original SD1.5 code) so the exact same fit can be run twice: once for the primary
        (per-token) embedding axis, and once, in parallel, for the pooled-embedding axis introduced
        for SDXL's `added_cond_kwargs`.

        Parameters:
            base_embeddings (np.ndarray): Array of shape (n, k) containing n summary vectors of
                dimensionality k.
        Returns:
            center (Tensor), radius (float), basis (Tensor) of shape (k, n-1).
        """
        n = base_embeddings.shape[0]  # n_embedding_axis
        k = base_embeddings.shape[-1]  # embedding dimension

        # Linear equations to compute the center of the circumscribed hypersphere.
        # Conditions: The center C (in the CLIP space) lies on the hyperplane spanned by the base_embeddings
        # (i.e. C can be (II) written as linear combination of some lambda_i of the base_embeddings
        # with (I) sum of lambda_i = 1)
        # and
        # all points (base_embeddings) have equal distance from the center
        # (i.e. (III) the same distance as the distance between base_embeddings_1 and the center).
        # Equation (III) can be written as: For each embedding_axis i, ||x_i - C||^2 = ||x_1 - C||^2, and thus
        # sum_j 2 (x_{1,j} - x_{i,j}) c_j = sum_j x_{1,j}^2 - sum_j x_{i,j}^2
        # The following system A x = b gives this solution with x = [lambda_1 ... lambda_n c_1 ... c_k].

        A = np.block([[np.ones([1, n]), np.zeros([1, k])],  # equation (I)
                      [base_embeddings.T, - np.eye(k)],
                      # equation (II), i.e. for each CLIP dimension: sum of lambda_i*x_i - c = 0
                      [np.zeros([n - 1, n]),
                       2 * (base_embeddings[0, np.newaxis] - base_embeddings[1:])]])  # equation (III)

        b = np.concatenate([np.ones([1]),  # equation (I)
                            np.zeros([k]),  # equation (II)
                            (np.sum(base_embeddings[0] ** 2, axis=-1, keepdims=True)
                             - np.sum(base_embeddings[1:] ** 2, axis=-1,
                                      keepdims=True)).flatten()])  # equation (III)

        C = np.linalg.solve(A, b)[-k:]  # discard solutions for lambda and only get the solution for C

        rel = base_embeddings - C  # move the base_embeddings by C so that their new center is 0 instead

        # get orthonormal basis -> any linear combination with coefficients that have the sum of squares of 1
        # will yield an admissible point on the circumscribed hypersphere
        Q_, _ = np.linalg.qr(rel.T)

        center = torch.Tensor(C)
        radius = np.linalg.norm(base_embeddings[0] - C)
        basis = torch.Tensor(Q_[:, :n - 1])  # discard one dimension since we are in a lower-dimensional user space

        return center, radius, basis

    def inv_transform(self, user_embeddings: Tensor):
        """
        This function transforms embeddings in the user_space back into the clip embedding space.

        Parameters:
            user_embeddings (Tensor): Parameters concerning the initially defined axis of a user_embedding.

        Returns
            clip_embeddings (Tensor): The respective clip embeddings.
            pooled_embeddings (Tensor): SDXL MIGRATION: new return value. The respective pooled
                embeddings, required by SDXL's `added_cond_kwargs`. See branch-level comments below
                for how each recommender type derives (or falls back to a constant for) this value.
            latents (Tensor): The respective latents.
        """
        if self.n_latent_axis:
            latent_factors = user_embeddings[:, -self.latent_axis.shape[0]:]
            user_embeddings = user_embeddings[:, :-self.latent_axis.shape[0]]

        # r = n_rec, a = n_axis, t = n_tokens, e = embedding_size
        if self.recommendation_type == RecommendationType.BASELINE:
            clip_embeddings = self.prompt_embedding.repeat(user_embeddings.shape[0], 1, 1)
            # SDXL MIGRATION: constant pooled-embedding fallback — this recommender type only
            # varies latents, not embeddings, so there's no coefficient scheme to interpolate a
            # matching pooled vector with; we just reuse the center prompt's pooled embedding for
            # every recommendation.
            pooled_embeddings = self.pooled_prompt_embedding.repeat(user_embeddings.shape[0], 1)
        elif self.recommendation_type in [RecommendationType.HYPERSPHERICAL_RANDOM,
                                          RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                          RecommendationType.HYPERSPHERICAL_BAYESIAN]:

            # we only have an orthonormal basis around the origin 0, so we need to scale by the radius of the
            # circumscribed hypersphere and translate to its center
            clip_embeddings = user_embeddings @ self.hyperspherical_basis.T * self.hyperspherical_radius + self.hyperspherical_center

            clip_embeddings = self.get_full_text_embeddings(clip_embeddings)

            clip_embeddings = slerp(clip_embeddings, self.prompt_embedding.repeat(user_embeddings.shape[0], 1, 1),
                                    self.original_prompt_share) * torch.linalg.norm(self.prompt_embedding, dim=-1,
                                                                                    keepdim=True)

            # SDXL MIGRATION: reconstruct the pooled embedding using the SAME per-recommendation
            # coefficients (`user_embeddings`) through the parallel pooled hyperspherical basis
            # fit in load_user_profile_host(), so it stays semantically consistent with the
            # primary embedding above. No `get_full_text_embeddings` step is needed here since
            # pooled vectors have no token dimension.
            pooled_embeddings = (user_embeddings @ self.hyperspherical_basis_pooled.T
                                 * self.hyperspherical_radius_pooled + self.hyperspherical_center_pooled)
            pooled_embeddings = slerp(pooled_embeddings,
                                     self.pooled_prompt_embedding.repeat(user_embeddings.shape[0], 1),
                                     self.original_prompt_share) * torch.linalg.norm(self.pooled_prompt_embedding,
                                                                                    dim=-1, keepdim=True)
        else:
            user_embeddings = user_embeddings.type(self.text_encoder.dtype)
            self.embedding_axis = self.embedding_axis.type(self.text_encoder.dtype)
            product = torch.einsum('ra,ate->rte', user_embeddings, self.embedding_axis)
            embedding_length = self.embedding_length.reshape((1, product.shape[1], 1))
            clip_embeddings = (self.embedding_center + product)
            clip_embeddings = (clip_embeddings / torch.linalg.vector_norm(clip_embeddings, ord=2, dim=-1, keepdim=True)
                               * embedding_length)
            # SDXL MIGRATION: constant pooled-embedding fallback, see BASELINE branch above — this
            # recommender family (RANDOM/EMA_DIRICHLET/etc.) doesn't have a hyperspherical basis to
            # reuse coefficients through, so a proper per-recommendation pooled vector isn't
            # derived here. Revisit if these recommender types end up being used for real runs.
            pooled_embeddings = self.pooled_prompt_embedding.repeat(user_embeddings.shape[0], 1)

        latents = None
        if self.recommendation_type in [RecommendationType.HYPERSPHERICAL_RANDOM,
                                        RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                        RecommendationType.HYPERSPHERICAL_BAYESIAN]:

            # no normalization required here since we ensured that the sum of squares of the latent_factors is one,
            # and thus we don't change the distribution parameters of the normal distribution
            latents = torch.einsum('rl,lxyz->rxyz', latent_factors, self.latent_axis)
        elif self.recommendation_type == RecommendationType.BASELINE:
            latents = latent_factors
        else:
            if self.n_latent_axis:
                latents = self.latent_center + torch.einsum('rl,lxyz->rxyz', latent_factors, self.latent_axis)
                latents = torch.nan_to_num(latents, nan=0.0)  # avoid SVD LinAlgError for all zero preferences
                latents = (latents / torch.linalg.matrix_norm(latents, ord=2, dim=(-2, -1), keepdim=True)
                           * self.latent_space_length)

        return clip_embeddings, pooled_embeddings, latents

    def fit_user_profile(self, preferences: Tensor):
        """
        This function initializes and fits a gaussian process for the available user preferences that can subsequently
        be used to generate new interesting embeddings for the user.

        Parameters:
            preferences (Tensor) : Preferences regarding the embeddings recommended last as real valued numbers.
        Returns:
            user_profile (Variable) : The fitted user profile depending on the optimizer.
        """
        # Initialize or extend the available user related data
        if self.preferences is not None:
            self.preferences = torch.cat((self.preferences, preferences))
        else:
            self.preferences = preferences

        # Only fit user profile if preferences are not all zero
        if torch.count_nonzero(self.preferences) > 0:
            if self.user_profile is not None:
                self.user_profile_history.append(self.user_profile)
            self.user_profile = self.optimizer.optimize_user_profile(self.embeddings, self.preferences,
                                                                     self.user_profile, self.beta)

    @torch.no_grad()
    def clip_embedding(self, prompt: str):
        """
        Embeds a given prompt using CLIP.

        Returns:
            sequence_embedding (Tensor): An embedding for the prompt in shape (77, 2048).
            pooled_embedding (Tensor): A (1280,) pooled/summary embedding for the prompt.

        SDXL MIGRATION: previously this called `self.stable_dif_pipe.encode_prompt(...)` once and
        returned its single (77, 768) sequence output. SDXL needs both encoders' per-token hidden
        states concatenated (matching what the SDXL UNet expects as `encoder_hidden_states`), plus
        a separate pooled vector for `added_cond_kwargs`. We tokenize/run both encoders manually
        (rather than using `encode_prompt`) so the pooled vector can use the summary-token
        extraction technique below instead of the encoder's own pooled/projected output.
        """

        def encode(tokenizer, text_encoder):
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt").to(text_encoder.device)
            outputs = text_encoder(text_inputs.input_ids, output_hidden_states=True)
            # penultimate hidden state, matching diffusers' own SDXL encode_prompt convention
            sequence_states = outputs.hidden_states[-2].cpu()
            # SDXL MIGRATION: summary-token extraction technique validated by the user in prior
            # (SD1.5-based) research — take the last *real* (non-padding, pre-EOS) token position,
            # i.e. the last token that received "fresh" real input, instead of blindly indexing the
            # final padded position or using the encoder's built-in pooled output. `attention_mask`
            # (not token id) is what marks real vs. padding tokens, so `attention_mask.sum()-1` is
            # the real EOS position and `attention_mask.sum()-2` is the last real content token
            # before it. Verified directly (2026-08-02) against both loaded tokenizers via
            # `tokenizer_2.decode(text_inputs.input_ids[0, summary_idx])` for short prompts and a
            # 100-token prompt that forces truncation — correctly resolves to the last real word in
            # all cases, not `<|endoftext|>` or padding, for both encoders. Note this holds despite
            # `tokenizer_2` (OpenCLIP-bigG) NOT sharing `tokenizer`'s pad_token==eos_token identity
            # (tokenizer_2.pad_token is `'!'`/id 0, distinct from its `<|endoftext|>` eos_token) —
            # the formula only depends on attention_mask correctly flagging real-vs-padded
            # positions, which holds for both tokenizers regardless of the specific pad token used.
            summary_idx = text_inputs.attention_mask.sum(dim=-1) - 2
            summary_token = outputs.hidden_states[-2][torch.arange(sequence_states.shape[0]), summary_idx].cpu()
            return sequence_states, summary_token

        sequence_1, _ = encode(self.tokenizer, self.text_encoder)
        sequence_2, summary_2 = encode(self.tokenizer_2, self.text_encoder_2)

        sequence_embedding = torch.cat([sequence_1, sequence_2], dim=-1)
        # SDXL MIGRATION: the pooled embedding is sourced from the second encoder (OpenCLIP-bigG,
        # 1280-dim) only, matching the dimensionality SDXL's added_cond_kwargs/pooled_prompt_embeds
        # expects (CLIP-L's 768-dim summary token would be the wrong shape).
        pooled_embedding = summary_2

        return sequence_embedding.squeeze(0), pooled_embedding.squeeze(0)

    def generate_recommendations(self, num_recommendations: int = 2):
        """
        This function generates recommendations based on the previously fit user-profile.

        Parameters:
            num_recommendations (int): Defines the number of embeddings that will be returned for user evaluation.
            beta (float): Trade-off between exploration and exploitation.
                Must be in [0, 1]. 0 means exploration, 1 means exploitation.
                Beta is increased after each recommendation (i.e. more exploitation).
                Optional, if given (by the debug menu), it will be used for the next generation of images.
        Returns:
            embeddings (Tensor): Embeddings that can be retransformed into the CLIP space and used for image generation
        """
        # The first recommender does not make use of the originally created subspace but dynamically generates new prompts based on user voting behavior
        if self.recommendation_type == RecommendationType.SIMPLE:
            # Extract probability distributions over all terms from user profile (None equals uniform distribution)
            if self.user_profile is not None:
                img_weights, sec_weights, at_weights, qual_weights, lat_weights = self.user_profile

                # Debug prints
                for weights, terms, name in zip([img_weights, sec_weights, at_weights, qual_weights],
                                                [self.image_styles, self.secondary_contexts,
                                                 self.atmospheric_attributes, self.quality_terms],
                                                ['Image Styles:', 'Secondary Contexts:', 'Atmospheric Attributes:',
                                                 'Quality Terms:']):
                    top_val, top_idx = torch.topk(torch.tensor([w for w in weights]), k=4)
                    print("Top 4 " + name)
                    for val, idx in zip(top_val, top_idx):
                        print(terms[idx], '[' + str(round(val.item() * 100, 2)) + '%]')
            else:
                img_weights, sec_weights, at_weights, qual_weights, lat_weights = None, None, None, None, None

            # Select new indices that build up the next embeddings
            img_idx = self.recommendation_prompt_generator.choices(range(len(self.image_styles)), weights=img_weights,
                                                                   k=num_recommendations)
            sec_idx = self.recommendation_prompt_generator.choices(range(len(self.secondary_contexts)),
                                                                   weights=sec_weights, k=num_recommendations)
            at_idx = self.recommendation_prompt_generator.choices(range(len(self.atmospheric_attributes)),
                                                                  weights=at_weights, k=num_recommendations)
            qual_idx = self.recommendation_prompt_generator.choices(range(len(self.quality_terms)),
                                                                    weights=qual_weights, k=num_recommendations)
            lat_idx = self.recommendation_prompt_generator.choices(range(self.n_latent_axis), weights=lat_weights,
                                                                   k=num_recommendations)

            # Generate respective clip embeddings (note that no inv-transformation is required here)
            print("The following prompts will be generated with various latents:")
            clip_embeddings = []
            # SDXL MIGRATION: parallel list of pooled embeddings — clip_embedding() now returns a
            # (sequence, pooled) tuple, see its docstring.
            pooled_embeddings = []
            for i in range(num_recommendations):
                prompt = self.image_styles[img_idx[i]] + self.original_prompt + self.secondary_contexts[sec_idx[i]] + \
                         self.atmospheric_attributes[at_idx[i]] + self.quality_terms[qual_idx[i]]
                print(str(i + 1) + ":", prompt)
                c_emb, p_emb = self.clip_embedding(prompt)
                clip_embeddings.append(c_emb)
                pooled_embeddings.append(p_emb)
            clip_embeddings = torch.stack(clip_embeddings)
            pooled_embeddings = torch.stack(pooled_embeddings)

            # For latents, simply select the respective latent from a list
            latents = self.latent_axis[lat_idx]

            # Update user profile
            if self.embeddings is not None:
                self.embeddings[0].extend(img_idx)
                self.embeddings[1].extend(sec_idx)
                self.embeddings[2].extend(at_idx)
                self.embeddings[3].extend(qual_idx)
                self.embeddings[4].extend(lat_idx)
            else:
                self.embeddings = [img_idx, sec_idx, at_idx, qual_idx, lat_idx]

        # This case works with the predefined user axis
        else:
            # Generate recommendations in the user_space
            if self.user_profile is not None or self.recommendation_type == RecommendationType.BASELINE:
                # obtain beta from the recommender if not given
                user_space_embeddings = self.recommender.recommend_embeddings(user_profile=self.user_profile,
                                                                              n_recommendations=num_recommendations,
                                                                              beta=self.beta)
            else:
                # Start initially with a lot of random embeddings to build a foundation for the user profile
                user_space_embeddings = self.random_recommender.recommend_embeddings(None, num_recommendations)

            # Transform embeddings from user_space to CLIP space
            clip_embeddings, pooled_embeddings, latents = self.inv_transform(user_space_embeddings)

            user_space_embeddings.type(self.text_encoder.dtype)
            # Safe the user_space_embeddings
            if self.embeddings is not None:
                self.embeddings = torch.cat((self.embeddings, user_space_embeddings))
            else:
                self.embeddings = user_space_embeddings

        # Update Beta and return clip embeddings and latents for a generator to use
        self.beta = min(self.beta + self.beta_step_size, 1.)
        return clip_embeddings, pooled_embeddings, latents

    def plotting_utils(self):
        """
        This function creates a reduction of the user embeddings into a two-dimensional space, so we can plot the
        embedding space and the respective images in our application.
        Parameters:
            algorithm (str) : Defines, which algorithm to use for the reduction.
        Returns:
            2D-user_profile (Tensor) : The user profile on which we base our recommendations on.
            2D-user_embeddings (Tensor) : Two dimensional reduction of the embeddings that resulted in the previously
                generated images
            Preferences (Tensor) : The respective preferences as a number between 0 and 1.
        """

        assert self.recommendation_type != RecommendationType.SIMPLE, "This is not yet available for the simple recommender."

        if self.num_axis == 2:
            return self.user_profile, self.embeddings, self.preferences

        else:
            # Check for GP-User Embedding
            if self.recommendation_type == RecommendationType.FUNCTION_BASED or self.recommendation_type == RecommendationType.RANDOM or self.recommendation_type == RecommendationType.DIVERSE_DIRICHLET:
                matrix = self.embeddings
                pca = PCA(n_components=2).fit(matrix)
                transformed_embeddings = pca.transform(matrix)

                if self.recommendation_type == RecommendationType.RANDOM or self.recommendation_type == RecommendationType.DIVERSE_DIRICHLET:
                    return None, transformed_embeddings, self.preferences

                # Retrieve scores for heatmap (function-based recommender)
                x = torch.linspace(-1, 1, 200)
                y = torch.linspace(-1, 1, 200)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                low_d_user_space = torch.cat((grid_x.flatten().reshape(-1, 1), grid_y.flatten().reshape(-1, 1)), dim=1)
                user_space = pca.inverse_transform(low_d_user_space).type(self.text_encoder.dtype)
                scores = self.recommender.heat_map_values(user_profile=self.user_profile,
                                                          user_space=user_space)
                if scores is not None:
                    scores = scores.reshape(grid_x.shape)

                return (x, y, scores), transformed_embeddings, self.preferences

            else:
                # First iteration, no user profile yet
                if self.user_profile is None:
                    matrix = self.embeddings

                else:
                    matrix = torch.cat((self.user_profile.reshape(1, -1), self.embeddings), dim=0)

                pca = PCA(n_components=2)
                transformed_embeddings = pca.fit_transform(matrix)

                if self.user_profile is None:
                    return None, transformed_embeddings, self.preferences

                else:
                    print(f'User profile history: {self.user_profile_history}')
                    low_d_user_profile = transformed_embeddings[0]
                    low_d_embeddings = transformed_embeddings[1:]
                    return low_d_user_profile, low_d_embeddings, self.preferences
