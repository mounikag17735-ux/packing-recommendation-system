import os
from flask import Flask

from analytics.db_setup import setup_database
from analytics.train_model import train_models


def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(base_dir)

    template_path = os.path.join(project_root, "templates")
    app = Flask(__name__, template_folder=template_path)

    # ---------------------------------------------------
    # HF FIRST BOOT SEQUENCE
    # ---------------------------------------------------

    print("🗄️ Setting up database...")
    setup_database()

    model_path = os.path.join(project_root, "analytics", "rf_cost_model.pkl")
    encoder_path = os.path.join(project_root, "analytics", "industry_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("🤖 Models not found. Training now...")
        train_models()
        print("✅ Models ready!")

    # ---------------------------------------------------

    from app.routes import main
    app.register_blueprint(main)

    print("🚀 App ready!")
    return app
