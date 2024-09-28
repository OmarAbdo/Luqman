from flask import Blueprint, jsonify
from app.services.hello_service import get_hello

hello_bp = Blueprint("hello", __name__)


@hello_bp.route("/", methods=["GET"])
def say_hello():
    message = get_hello()
    return jsonify({"message": message}), 200
