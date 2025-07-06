import flask
from .routes import init_routes

def create_app():
    app = flask(__name__)
    init_routes(app)
    return app