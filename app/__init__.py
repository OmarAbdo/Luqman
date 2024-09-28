from flask import Flask


def create_app():
    app = Flask(__name__)

    # Register Blueprints
    from app.controllers.hello_controller import hello_bp

    app.register_blueprint(hello_bp, url_prefix="/api/hello")

    from app.controllers.stock_controller import stock_bp

    app.register_blueprint(stock_bp, url_prefix="/api/stock")

    return app
