from nicegui import ui as ngUI

from prototype.constants import WebUIState, RecommendationType, ScoreMode

class DebugMenu(ngUI.element):
    """
    Contains the code for the debug menu.
    """

    def __init__(self, webUI, bg_color=(0,0,0,0.5)):
        """
        Builds the debug menu UI.

        Args:
            webUI: The parent WebUI instance that uses this component.
            bg_color: The background color of the debug menu. Defaults to (0,0,0,0.5).
        """
        super().__init__(tag='div')
        self.webUI = webUI
        self.style('position: fixed; display: block; width: 100%; height: 100%;'
                   'top: 0; left: 0; right: 0; bottom: 0; z-index: 2;'
                   'background-color:' + f'rgba{bg_color};')
        self.input_props = 'input-style="color: white" label-color=white dense'
        self.checkbox_style = 'color: white;'
        with self:
            with ngUI.element('div').style("margin-left: 8px;").classes("h-screen flex"):
                with ngUI.column().classes('p-0 gap-0'):
                    ngUI.label("Debug UI Info").style("font-size: 50px; color: white;")
                    ngUI.input(label="session_id").props(self.input_props).bind_value(self.webUI, 'session_id')
                    ngUI.select({s: s.value for s in WebUIState}, label='state', on_change=self.webUI.update_state_variables).props(self.input_props).bind_value(self.webUI, 'state')
                    ngUI.checkbox('is_initial_iteration').style(self.checkbox_style).bind_value(self.webUI, 'is_initial_iteration')
                    ngUI.checkbox('is_main_loop_iteration').style(self.checkbox_style).bind_value(self.webUI, 'is_main_loop_iteration')
                    ngUI.checkbox('is_generating').style(self.checkbox_style).bind_value(self.webUI, 'is_generating')
                    ngUI.checkbox('is_interactive_plot').style(self.checkbox_style).bind_value(self.webUI, 'is_interactive_plot')
                    ngUI.input(label="user_prompt").props(self.input_props).bind_value(self.webUI, 'user_prompt')
                    ngUI.select({t: t.value for t in RecommendationType}, label='recommendation_type').props(self.input_props).bind_value(self.webUI, 'recommendation_type')
                    ngUI.number(label="num_images_to_generate", min=0, precision=0, step=1).props(self.input_props).bind_value(self.webUI, 'num_images_to_generate', forward=int)
                    ngUI.select({m.value: m.value for m in ScoreMode}, label='score_mode').style("width: 100px;").props(self.input_props).bind_value(self.webUI, 'score_mode')
                    ngUI.number(label="image_display_width", min=0, precision=0, step=1).props(self.input_props).bind_value(self.webUI, 'image_display_width', forward=int)
                    ngUI.number(label="image_display_height", min=0, precision=0, step=1).props(self.input_props).bind_value(self.webUI, 'image_display_height', forward=int)
                    ngUI.number(label="active_image", min=0, precision=0, step=1).props(self.input_props).bind_value(self.webUI, 'active_image', forward=int)
                    ngUI.input(label="save_path").props(self.input_props).bind_value(self.webUI, 'save_path')
                    ngUI.button('Force UI Reload', on_click=self.webUI.reload_userinterface, color='red').style("margin-top: 40px; margin-left: 10px;")
                ngUI.space()
                self.user_profile_host_info = ngUI.column().classes('p-0 gap-0')
                with self.user_profile_host_info:
                    ngUI.label("Debug UP Host Info").style("font-size: 50px; color: white;")
                    ngUI.label("Not initialized").style("font-size: 20px; color: red;").bind_visibility(self.webUI, "is_initial_iteration", value=True)
                    # This weird structure is a workaround for a nicegui bug
                    self.up_info = [
                    ("original_prompt", ngUI.input(label="original_prompt").props(self.input_props)),
                    ("recommendation_type", ngUI.select({t: t.value for t in RecommendationType}, label='recommendation_type').props(self.input_props)),
                    ("height", ngUI.number(label="height", min=0, precision=0, step=8).props(self.input_props)),
                    ("width", ngUI.number(label="width", min=0, precision=0, step=8).props(self.input_props)),
                    ("latent_space_length", ngUI.number(label="latent_space_length", min=0, step=0.01).props(self.input_props)),
                    ("n_latent_axis", ngUI.number(label="n_latent_axis", min=0, precision=0, step=1).props(self.input_props)),
                    ("n_embedding_axis", ngUI.number(label="n_embedding_axis", min=0, precision=0, step=1).props(self.input_props)),
                    ("use_embedding_center", ngUI.checkbox('use_embedding_center').style(self.checkbox_style)),
                    ("use_latent_center", ngUI.checkbox('use_latent_center').style(self.checkbox_style)),
                    ("n_recommendations", ngUI.number(label="n_recommendations", min=0, precision=0, step=1).props(self.input_props)),
                    ("ema_alpha", ngUI.number(label="ema_alpha", min=0, step=0.01).props(self.input_props)),
                    ("beta", ngUI.number(label="beta", min=0, precision=2, step=.01).props(self.input_props)),
                    ("beta_step_size", ngUI.number(label="beta_step_size", min=0, precision=2, step=.01).props(self.input_props)),
                    ("local_search_fraction", ngUI.number(label="local_search_fraction", min=0, max=1, precision=2, step=.01).props(self.input_props)),
                    ]
        self.toggle_visibility()

    def set_user_profile_updater(self):
        """
        Sets the function and bindings of the user profile host debug info.
        """
        for attribute, ui_element in self.up_info:
            ui_element.bind_value(self.webUI.user_profile_host, attribute)
            if attribute not in ["beta", "beta_step_size", "local_search_fraction"]:
                ui_element.on_value_change(self.webUI.user_profile_host.load_user_profile_host)
    
    def toggle_visibility(self):
        """
        Toggles visibility of the debug menu.
        """
        self.set_visibility(not self.visible)
