from flask import Flask

class WebInterface:

    def __init__(self, name):
        # Trainer
        self.trainer = None
        # Flask App
        self.app = Flask(name, static_url_path='/static', template_folder='templates/')
        return

    def add_endpoint(self):

        return

    def setup_routes(self):

        return


