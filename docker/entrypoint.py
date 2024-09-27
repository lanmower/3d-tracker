import os
import subprocess
import json
import requests
import uuid
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Define output directories
PRED_RESULTS_DIR = "vis_results"

# Set up your webhook URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec")

# Global ThreadPoolExecutor for managing inference tasks
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    subprocess.run(command, check=True)

def process_image(image_url):
    try:
        # Generate a random filename to prevent collisions
        random_filename = str(uuid.uuid4())
        output_dir = os.path.join(PRED_RESULTS_DIR, random_filename)

        # Create output directory for this processing
        os.makedirs(output_dir, exist_ok=True)

        # Define the inference commands
        command = [
            "python", "body3d_img2pose_demo.py",
            "rtmdet_m_640-8xb32_coco-person.py",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
            "rtmw3d-l_cock14-0d4ad840_20240422.pth", "--save-predictions", "--input", image_url,
            "--output-root", output_dir,
        ]

        # Run inference command
        run_inference(command)

        # Get the latest prediction JSON file
        latest_json = get_latest_prediction(output_dir)

        # If a JSON file exists, send it via webhook
        if latest_json:
            call_webhook(latest_json)
        else:
            print("No JSON file found for webhook.")

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))

def get_latest_prediction(output_dir):
    """Get the latest prediction JSON file."""
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    if not json_files:
        return None
    latest_json_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
    with open(os.path.join(output_dir, latest_json_file), "r") as json_file:
        return json.load(json_file)

def call_webhook(data):
    """Send the processed data to the webhook."""
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 200:
            print("Webhook sent successfully:", response.status_code, response.text)
        else:
            print("Webhook response error:", response.status_code, response.text)
    except Exception as e:
        print("Error calling webhook:", str(e))

@app.route('/process_image', methods=['GET'])
def handle_process_image():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "image_url parameter is required"}), 400

    # Start processing the image
    executor.submit(process_image, image_url)
    return jsonify({"message": "Image processing started"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
