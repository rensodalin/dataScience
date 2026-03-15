# app/app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    file = request.files["image"]
    file.save("temp_query.jpg")
    results = search_similar("temp_query.jpg", top_k=3)
    
    # Add reviews from your database
    for r in results:
        r["reviews"] = get_reviews(r["product"])  # from your reviews_db.json
    
    return jsonify(results)