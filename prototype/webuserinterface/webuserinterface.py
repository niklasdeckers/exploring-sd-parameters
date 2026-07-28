from nicegui import ui as ngUI
from nicegui import binding
from nicegui.events import KeyEventArguments
from PIL import Image
import asyncio
import threading
import secrets
import base64
from io import BytesIO

from prototype.constants import RecommendationType, WebUIState, ScoreMode
from prototype.user_profile_host import UserProfileHost
from prototype.utils import seed_everything
from prototype.webuserinterface.components import InitialIterationUI, MainLoopUI, LoadingUI, PlotUI, Scorer, DebugMenu


class WebUI:
    """
    This class implements a interactive web user interface for an image generation system.
    """
    session_id = binding.BindableProperty()
    state = binding.BindableProperty()
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()
    is_interactive_plot = binding.BindableProperty()
    iteration = binding.BindableProperty()
    user_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()
    num_images_to_generate = binding.BindableProperty()
    score_mode = binding.BindableProperty()
    image_display_width = binding.BindableProperty()
    image_display_height = binding.BindableProperty()
    active_image = binding.BindableProperty()
    save_path = binding.BindableProperty()
    blind_mode = binding.BindableProperty()

    @classmethod
    async def create(cls, args, pipe, generator, queue_lock):
        """
        This method should be used instead of the __init__-method to create an object of the WebUI-class.
        Usage: ui = await WebUI.create(...) inside an async function.

        Args:
            args: The config args as an omegaconf.DictConfig object.
            pipe: The central SD-pipeline used for the generator.

        Returns:
            Created object of type WebUI.
        """
        self = cls()
        loading_label = ngUI.label("Starting session...")
        await ngUI.context.client.connected()
        # Args of global config
        self.args = args
        self.pipe = pipe
        self.generator = generator
        seed_everything(self.args.random_seed)
        self.queue_lock = queue_lock
        # Generate id for this session
        self.session_id = secrets.token_urlsafe(4)
        # State variables
        self.state = None
        self.is_initial_iteration = False
        self.is_main_loop_iteration = False
        self.is_generating = False
        self.is_interactive_plot = False
        self.iteration = 0
        # Provided by the user / system
        self.user_prompt = ""
        self.recommendation_type = RecommendationType.HYPERSPHERICAL_BAYESIAN
        self.num_images_to_generate = self.args.num_recommendations
        self.first_iteration_images_factor = self.args.first_iteration_images_factor

        self.score_mode = self.args.score_mode
        self.scorer = Scorer(self)

        # Other modules
        self.user_profile_host = None # Initialized after initial iteration

        # Lists / UI components
        self.image_display_width, self.image_display_height = tuple(self.args.image_display_size)
        self.images = [Image.new('RGB', (self.image_display_width, self.image_display_height)) for _ in range(self.num_images_to_generate * self.args.first_iteration_images_factor)] # For convenience already initialized here
        self.images_display = [None for _ in range(self.num_images_to_generate * self.args.first_iteration_images_factor)] # For convenience already initialized here
        self.active_image = 0
        self.submit_button = None
        # Image saving
        self.save_path = f"{self.args.path.images_save_dir}/{self.session_id}"
        self.num_images_saved = 0

        self.blind_mode = False

        # Set UI root & load debug menu
        self.setup_root()
        self.debug_menu = DebugMenu(self)

        self.keyboard = ngUI.keyboard(on_key=self.handle_key)
        # Remove loading label
        loading_label.delete()
        loading_label = None
        return self

    def run(self):
        """
        This function starts the Web UI.
        """
        print("Start running the Web UI.")
        self.change_state(WebUIState.INIT_STATE)
        self.root.clear()
        self.build_userinterface()

    def reload_userinterface(self):
        """
        Reloads the UI.
        """
        self.root.clear()
        self.scorer = Scorer(self)
        self.images = self.images[:min(len(self.images), self.num_images_to_generate * self.args.first_iteration_images_factor)] \
                    + [Image.new('RGB', (self.image_display_width, self.image_display_height)) for _ in range((self.num_images_to_generate * self.args.first_iteration_images_factor) - min(len(self.images), self.num_images_to_generate * self.args.first_iteration_images_factor))]
        self.images_display = [None for _ in range(self.num_images_to_generate * self.args.first_iteration_images_factor)]
        self.build_userinterface()

    # <---------- Updating State ---------->
    def change_state(self, new_state: WebUIState):
        """
        Updates the current state of the Web UI.

        Args:
            new_state: The updated state of the Web UI.
        """
        self.state = new_state
        self.update_state_variables()

    def update_state_variables(self):
        """
        Updates the boolean state variables (used for component visibility) based on the current state of the web UI.
        """
        self.is_initial_iteration = self.state == WebUIState.INIT_STATE
        self.is_main_loop_iteration = self.state == WebUIState.MAIN_STATE
        self.is_generating = self.state == WebUIState.GENERATING_STATE
        self.is_interactive_plot = self.state == WebUIState.PLOT_STATE       


    # <------------------------------------>
    # <---------- Building UI ---------->
    def build_userinterface(self):
        """
        Builds the complete user interface using NiceGUI.

        UI Structure:
        - Webis demo template top half.
        - Content based on the current state. Either the initial prompt input, the main loop with the user preferences, the loading spinner or the plot.
        - Some empty space so the footer doesnt look weird on high resolution devices.
        - Webis demo template bottom half/footer.
        """
        print("Building User Interface.")
        webis_template_top, webis_template_bottom = self.get_webis_demo_template_html()
        with self.root:
            ngUI.html(webis_template_top).classes('w-full')
            ngUI.space().classes('w-full h-full')
            InitialIterationUI(self)
            self.main_loop_ui = MainLoopUI(self)
            self.loading_ui = LoadingUI(self)
            self.plot_ui = PlotUI(self)
            ngUI.space().classes('w-full h-full')
            ngUI.html(webis_template_bottom).classes('w-full')

    def setup_root(self):
        """
        Setups the root element, where all the other UI elements will be placed.
        """
        self.root = ngUI.column().classes('w-full h-full').style('font-family:"Product Sans","Noto Sans","Verdana", sans-serif;')
        ngUI.add_head_html('''
        <style>
        .nicegui-content {
            padding: 0;
        }
        </style>
        ''')
        ngUI.query('.nicegui-content').classes('w-full')
        ngUI.query('.q-page').classes('flex')

    # <--------------------------------->
    # <---------- Initialize other non-UI components ---------->

    def init_user_profile_host(self):
        """
        Initializes the user profile host with the initial user prompt.
        """
        print("Initialize User Profile Host.")
        self.user_profile_host = UserProfileHost(
            original_prompt=self.user_prompt,
            add_ons=None,
            recommendation_type=self.recommendation_type,
            cache_dir=self.args.path.cache_dir,
            stable_dif_pipe=self.generator.pipe,
            n_recommendations=self.num_images_to_generate,
            hf_model_name=self.args.hf_model_name,
            **self.args.recommender
        )

    # <------------------------------------------------------->
    # <---------- Keyboard controls ---------->
    def handle_key(self, e: KeyEventArguments):
        """
        Handles key events.

        Args:
            e: KeyEvent args.
        """
        if e.key.f9 and e.action.keydown:
            self.debug_menu.toggle_visibility()
        if self.score_mode == ScoreMode.EMOJI.value and self.state == WebUIState.MAIN_STATE:
            if e.key.location != 3 and e.key.arrow_right and e.action.keydown:
                self.update_active_image(self.active_image + 1)
            if e.key.location != 3 and e.key.arrow_left and e.action.keydown:
                self.update_active_image(self.active_image - 1)
            #if e.key == 's' and e.action.keydown:
            #    self.main_loop_ui.on_save_button_click(self.images_display[self.active_image])
            if e.key.enter and e.action.keydown:
                self.submit_button.run_method('click')
            if e.key.number in [1, 2, 3, 4, 5] and e.action.keydown:
                self.on_number_keystroke(e.key.number)
            if e.key.location == 3 and e.key.code in [f'Numpad{i}' for i in range(1, 1+5)] and e.action.keydown:
                self.on_number_keystroke(int(e.key.code[-1]))

    def update_active_image(self, idx=0):
        """
        Updates the active image and its visuals on the UI (currently only used in emoji ScoreMode).

        Args:
            idx: The image index of the new active image.
        """
        if self.score_mode == ScoreMode.EMOJI.value:
            idx = idx % len(self.images)
            self.images_display[self.active_image].style(f'width: {self.image_display_width}px; height: {self.image_display_height}px;')
            self.active_image = idx
            self.images_display[idx].style(f'width: {int(self.image_display_width*1.1)}px; height: {int(self.image_display_height*1.1)}px;')

    def on_number_keystroke(self, key):
        """
        Updates the score for the active image upon typing one of the valid number keys.

        Args:
            key: The number of the key typed.
        """
        self.scorer.scores_toggles[self.active_image].value = key - 1
        self.update_active_image(self.active_image + 1)

    # <--------------------------------------->
    # <---------- Image generation & User profile ---------->
    def generate_images(self):
        """
        Generates images by passing the recommended embeddings from the user profile host to the generator and saving the generated 
        images of the generator in self.images.
        """
        print("Generate new Images.")
        # SDXL MIGRATION: generate_recommendations() now returns an extra pooled_embeddings tensor
        # (required by SDXL's added_cond_kwargs), threaded straight through to generate_image().
        if self.iteration < 2:
            embeddings, pooled_embeddings, latents = self.user_profile_host.generate_recommendations(num_recommendations=self.num_images_to_generate*self.first_iteration_images_factor)
        else:
            embeddings, pooled_embeddings, latents = self.user_profile_host.generate_recommendations(num_recommendations=self.num_images_to_generate)
        self.images = self.generator.generate_image(embeddings, pooled_embeddings, latents, self.loading_ui.loading_progress, self.queue_lock)

    def update_image_displays(self):
        """
        Updates the image displays with the current images in self.images.
        """
        def jpg(img):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode(encoding="utf-8")
            return img_str
        print("Update Image Displays.")
        [self.images_display[i].set_source(jpg(self.images[i])) for i in range(len(self.images))]

    def update_user_profile(self):
        """
        Call the user profile host to update the user profile using provided scores of the current iteration.
        """
        print("Update UserProfileHost.")
        if self.iteration < 2:
            normalized_scores = self.scorer.get_scores()
        else:
            normalized_scores = self.scorer.get_scores()[:self.num_images_to_generate]
        self.user_profile_host.fit_user_profile(preferences=normalized_scores)

    # <----------------------------------------------------->
    # <---------- Misc. ---------->
    def get_webis_demo_template_html(self):
        """
        Returns the webis html template for demo web applications.

        Returns:
            A tuple of the top half of the webis html template until the demo content and the bottom half/footer.
        """
        with open("./prototype/resources/webis_template_top.html") as f:
            webis_template_top = f.read()
        with open("./prototype/resources/webis_template_bottom.html") as f:
            webis_template_bottom = f.read()
        return webis_template_top, webis_template_bottom

    # <--------------------------->
