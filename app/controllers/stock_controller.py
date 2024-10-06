from flask import Blueprint, jsonify, request
from app.services.stock_service import get_stock_data

stock_bp = Blueprint("stock", __name__)


@stock_bp.route("/", methods=["GET"])
def fetch_stock_data():
    ticker_symbol = request.args.get("ticker") 
    if not ticker_symbol:
        return jsonify({"error": "ticker query argument is required"}), 400
    stock_data = get_stock_data(ticker_symbol)
    return jsonify(stock_data), 200
