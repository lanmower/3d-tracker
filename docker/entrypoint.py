import os
import subprocess
import json
import requests
import uuid
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

app = Flask(__name__)

# Set up your webhook URL
WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL",
    "https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec",
)
print('starting flask')

# Global ThreadPoolExecutor for managing inference tasks
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    subprocess.run(command, check=True)

def process_image(image_url, task_id):
    # Define unique directories for this task
    base_output_dir = Path("vis_results") / task_id
    pred_results_dir = base_output_dir / "predictions"

    # Create output directories
    pred_results_dir.mkdir(parents=True, exist_ok=True)

    json_output_file = pred_results_dir / f"results.json"

    try:
        # Define the inference commands without visualization
        commands = [
            [
                "python",
                "body3d_img2pose_demo.py",
                "rtmdet_m_640-8xb32_coco-person.py",
                "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
                "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
                "rtmw3d-l_cock14-0d4ad840_20240422.pth",
                "--disable-rebase-keypoint",
                "--disable-norm-pose-2d",
                "--save-predictions",
                "--input",
                image_url,
                "--output-root",
                str(pred_results_dir),
                "--show",  # Remove or set to False to disable visualization
                "False"    # Explicitly disable show argument
            ]
        ]

        # Submit inference commands to executor
        futures = [executor.submit(run_inference, cmd) for cmd in commands]

        # Wait for all inference tasks to complete
        wait(futures)

        # Return the path of the generated JSON predictions file
        return str(json_output_file)

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))
        return None

@app.route("/process_image", methods=["GET"])
def handle_process_image():
    image_url = request.args.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url parameter is required"}), 400

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Start processing the image and get the JSON output file path
    json_file_path = executor.submit(process_image, image_url, task_id).result()

    if json_file_path:
        return jsonify({"message": "Image processing completed", "result_file": json_file_path}), 202
    else:
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == "__main__":
    # Ensure base directories exist
    Path("vis_results").mkdir(exist_ok=True)

    app.run(host="0.0.0.0", port=8080)
