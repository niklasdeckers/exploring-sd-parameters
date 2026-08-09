from nicegui import ui as ngUI
import asyncio
import base64
from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import WebUIState, ScoreMode
from prototype.utils import seed_everything


class MainLoopUI(UIComponent):
    """
    Contains the code for the main loop UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the main loop iteration state.
        """
        ngUI.html('<style>.multi-line-notification { white-space: pre-line; }</style>')
        with ngUI.column().classes('mx-auto items-center pl-24 pr-24').bind_visibility_from(self.webUI, 'is_main_loop_iteration', value=True):
            #with ngUI.row().classes('w-full justify-end mb-8'):
            #    ngUI.button('Interactive plot', icon='o_scatter_plot', on_click=self.on_show_interactive_plot_button_click).style('font-weight: bold;').props('color=secondary unelevated rounded')
            with ngUI.row().classes('mx-auto items-center'):
                ngUI.label('Please rate these images according to your preferences.').style('font-size: 200%;')
                if self.webUI.score_mode == ScoreMode.EMOJI.value:
                    ngUI.button(icon='o_info', on_click=lambda: ngUI.notify(
                        'Keyboard Controls:\n'
                        'Left/Right arrow: Navigate through images\n'
                        '1-5: Score current image\n'
                        #'s: Save current image\n'
                        'Enter: Submit scores',
                        multi_line=True,
                        classes='multi-line-notification'
                    )).props('flat fab color=black')
            with ngUI.column().classes('mx-auto items-center'):
                with ngUI.row().classes('w-full items-center justify-start').bind_visibility_from(self.webUI, 'blind_mode', value=False):
                    ngUI.icon('settings_suggest', size='2rem').classes('mr-2')
                    ngUI.label(self.webUI.recommendation_type).style('font-size: 120%;').bind_text_from(self.webUI, 'recommendation_type')
                with ngUI.row().classes('w-full items-center justify-start'):
                    ngUI.icon('subject', size='2rem').classes('mr-2')
                    ngUI.label(self.webUI.user_prompt).style('font-size: 120%;').bind_text_from(self.webUI, 'user_prompt')
            with ngUI.row().classes('mx-auto items-center mt-4'):
                for i in range(self.webUI.num_images_to_generate * self.webUI.first_iteration_images_factor):
                    with ngUI.column().classes('mx-auto items-center') as image_container:
                        if i >= self.webUI.num_images_to_generate:
                            image_container.bind_visibility_from(self.webUI, 'iteration', backward=lambda it: it < 2, value=True)
                        self.webUI.images_display[i] = ngUI.interactive_image().style(f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down;')
                        #with self.webUI.images_display[i]:
                        #    ngUI.button(icon='o_save', on_click=partial(self.on_save_button_click, self.webUI.images_display[i])).props('flat fab color=white').classes('absolute bottom-0 right-0 m-2')
                        self.webUI.scorer.build_scorer(i)
            ngUI.space()
            with ngUI.column().classes('w-full m-8'):
                with ngUI.row().classes('w-full items-center'):
                    ngUI.icon('explore')
                    ngUI.label('Exploration')
                    ngUI.space()
                    ngUI.label('Exploitation')
                    ngUI.icon('emoji_events')
                self.beta_slider = ngUI.slider(min=0, max=1, step=0.01).props('color=secondary label')
                if self.webUI.user_profile_host is not None:
                    self.set_user_profile_host_beta_updater()
            ngUI.space()
            with ngUI.row().classes('w-full'):
                ngUI.button('Restart process', icon='restart_alt', on_click=self.on_restart_process_button_click, color='red').style('font-weight: bold;').props('unelevated rounded')
                ngUI.space()
                self.webUI.submit_button = ngUI.button('Submit scores', on_click=self.on_submit_scores_button_click).style('font-weight: bold;').props('icon-right="navigate_next" color=grey-8 unelevated rounded')
    
    def on_show_interactive_plot_button_click(self):
        """
        Shows the interactive plot screen.
        """
        self.webUI.change_state(WebUIState.PLOT_STATE)

        # Initialize or update the plot
        self.webUI.plot_ui.update_view()
    
    def on_save_button_click(self, image_display):
        """
        Saves the displayed image where the save button is located in the images save dir.

        Args:
            image_display: The image display containing the image to save.
        """
        # image_display.source is a "data:image/jpeg;base64,..." data URL (see WebUI.update_image_displays'
        # jpg() helper), not a PIL Image, so it has to be decoded before it can be saved.
        _, encoded = image_display.source.split(",", 1)
        image_to_save = Image.open(BytesIO(base64.b64decode(encoded)))

        # self.webUI.save_path is user-editable via the debug menu, so it must be confined to the
        # configured images save directory to avoid writing to an arbitrary path on the server.
        base_dir = Path(self.webUI.args.path.images_save_dir).resolve()
        save_dir = Path(self.webUI.save_path).resolve()
        if save_dir != base_dir and base_dir not in save_dir.parents:
            ngUI.notify("Invalid save path: must be inside the images save directory.", type='negative')
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"image_{self.webUI.num_images_saved}.png"
        image_to_save.save(file_path)
        self.webUI.num_images_saved += 1
        ngUI.notify(f"Image saved in {file_path}!")

    async def on_submit_scores_button_click(self):
        """
        Updates the user profile with the user scores and generates the next images.
        """
        self.webUI.update_user_profile()
        ngUI.notify('Scores submitted!')
        ngUI.notify('Number of images updated!')
        self.webUI.change_state(WebUIState.GENERATING_STATE)
        self.webUI.iteration += 1
        ngUI.notify('Generating new images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.webUI.generate_images)
        self.webUI.update_image_displays()
        self.webUI.scorer.reset_scorers()
        self.webUI.change_state(WebUIState.MAIN_STATE)
        self.webUI.loading_ui.reset_progress_bar()
        self.webUI.update_active_image()

    def on_restart_process_button_click(self):
        """
        Restarts the process by starting with the initial iteration again.
        """
        self.webUI.change_state(WebUIState.INIT_STATE)
        self.webUI.scorer.reset_scorers()
        self.webUI.user_profile_host = None
        self.webUI.iteration = 0

        # Clear plot ui for new process
        self.webUI.generator.clear_latest_images()
        self.webUI.plot_ui.clear_view()

        seed_everything(self.webUI.args.random_seed)
        
    
    def set_user_profile_host_beta_updater(self):
        """
        Sets the value binding of the beta_slider.
        """
        self.beta_slider.bind_value(self.webUI.user_profile_host, 'beta', backward=lambda x: round(x, 2))
