from flask import Blueprint, jsonify, request

stock_bp = Blueprint("stock", __name__)


@stock_bp.route("/", methods=["GET"])
def fetch_stock_data():
    ticker_symbol = request.args.get("ticker") 
    if not ticker_symbol:
        return jsonify({"error": "ticker query argument is required"}), 400
    return "To be implemented", 200
