import os
from flask import Flask

from analytics.train_model import train_models


def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(base_dir)
    template_path = os.path.join(project_root, "templates")

    app = Flask(__name__, template_folder=template_path)

    from app.routes import main
    app.register_blueprint(main)

    return app