from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import tempfile
import os
import logging
import time
import psutil
from color import quantize_colors
from generate_stl import generate_stl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Starting app initialization")
start_time = time.time()

app = Flask(__name__, static_folder="client/build")
CORS(app)

logger.debug(f"Flask app initialized in {time.time() - start_time:.2f} seconds")


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/static-stl", methods=["GET"])
def get_static_stl():
    stl_file_path = "../client/public/output.stl"
    if os.path.exists(stl_file_path):
        return send_file(stl_file_path, as_attachment=True)
    else:
        return jsonify({"error": "STL file not found"}), 404


@app.route("/quantize-colors", methods=["POST"])
def quantize_image_colors():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get the selected colors from the request
    selected_colors = request.form.get("selected_colors", "")
    if not selected_colors:
        return jsonify({"error": "No colors selected"}), 400

    selected_colors = [
        tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        for color in selected_colors.split(",")
    ]

    remap_colors = request.form.get("remap_colors", "true").lower() == "true"
    reverse_palette = request.form.get("reverse_palette", "false").lower() == "true"

    # Read the image
    img = Image.open(file.stream)

    # Quantize colors
    quantized_img, color_palette = quantize_colors(
        img, selected_colors, remap_colors_to_palette=remap_colors, reverse_palette=reverse_palette
    )

    # Convert the quantized image to base64
    buffered = io.BytesIO()
    quantized_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"quantized_image": img_str, "color_palette": color_palette})


@app.route("/generate-stl", methods=["POST"])
def generate_stl_from_heightmap_api():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get parameters from the request
    color_palette = request.form.get("color_palette")
    if not color_palette:
        return jsonify({"error": "Color palette not provided"}), 400

    try:
        color_palette = [
            tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
            for h in color_palette.split(",")
        ]
    except ValueError:
        return jsonify({"error": "Invalid color palette format"}), 400

    object_height = float(request.form.get("object_height", 40))
    object_width = float(request.form.get("object_width", 70))

    # Read the image
    img = Image.open(file.stream)

    # Convert to numpy array
    height_map_image = np.array(img)

    # Generate STL
    stl_mesh = generate_stl(
        height_map_image,
        color_palette,
        object_height=object_height,
        object_width=object_width,
    )

    # Save STL to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        stl_mesh.export(tmp_file.name)

        # Read the temporary file and encode to base64
        with open(tmp_file.name, "rb") as f:
            stl_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Remove the temporary file
    os.unlink(tmp_file.name)

    return jsonify({"stlFile": stl_base64})


logger.debug(f"Total app initialization time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
