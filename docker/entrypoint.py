import os
import subprocess
import json
import requests
import uuid
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
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
    pose3d_output_dir = base_output_dir / "pose3d"
    pred_results_dir = base_output_dir / "predictions"

    # Create output directories
    pose3d_output_dir.mkdir(parents=True, exist_ok=True)
    pred_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Define the inference commands
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
                str(pose3d_output_dir),
            ]
        ]

        # Submit inference commands to executor
        for cmd in commands:
            executor.submit(run_inference, cmd)

        # Make sure enough time for the processes to finish before checking the output
        executor.shutdown(wait=True)

        # Get the specific JSON output file generated in the prediction directory
        output_file = pred_results_dir / "specific_output.json"  # Adjust this to your specific expected output file name

        # Process results from the specific JSON prediction file
        if output_file.is_file():
            combined_results = combine_and_convert_results([output_file])
            print("Combined Results:", combined_results)

            # Call webhook with results
            call_webhook(combined_results)

        else:
            print(f"No output file found at: {output_file}")

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))

def combine_and_convert_results(output_files):
    """Combine and process the JSON prediction files."""
    combined_data = {}

    # Process only JSON output files
    for file in output_files:
        if file.is_file() and file.suffix.lower() == ".json":
            with open(file, "r") as json_file:
                data = json.load(json_file)
                base_filename = file.stem
                print(f"Read data from {file}: {data}")  # Debug print for each file
                combined_data[base_filename] = data  # Capture the prediction data

    return combined_data  # Return the combined prediction data

def call_webhook(data):
    """Send the processed data to the webhook."""
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        print("Webhook response:", response.status_code, response.text)
    except Exception as e:
        print("Error calling webhook:", str(e))

@app.route("/process_image", methods=["GET"])
def handle_process_image():
    image_url = request.args.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url parameter is required"}), 400

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Start processing the image
    executor.submit(process_image, image_url, task_id)
    
    return jsonify({"message": "Image processing started", "task_id": task_id}), 202

if __name__ == "__main__":
    # Ensure base directories exist
    Path("vis_results").mkdir(exist_ok=True)

    app.run(host="0.0.0.0", port=8080)
