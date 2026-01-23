import os
from flask import Flask

from analytics.train_model import train_models


def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(base_dir)

    template_path = os.path.join(project_root, "templates")

    app = Flask(__name__, template_folder=template_path)

    # 🔥 HuggingFace first boot — train models if not present
    model_path = os.path.join(project_root, "analytics", "rf_cost_model.pkl")
    encoder_path = os.path.join(project_root, "analytics", "industry_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("🔧 Models not found. Training now...")
        train_models()
        print("✅ Models ready.")

    from app.routes import main
    app.register_blueprint(main)

    return app
