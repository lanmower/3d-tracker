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

        # Generate a unique name for the results JSON
        json_filename = f'results_{random_filename}.json'
        json_file_path = os.path.join(output_dir, json_filename)

        # Rename the old JSON file to the new unique name
        for f in os.listdir(output_dir):
            if f.endswith('.json'):
                old_json_file_path = os.path.join(output_dir, f)
                os.rename(old_json_file_path, json_file_path)
                print(f"Renamed JSON file to: {json_filename}")
                break

        # Load the renamed JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            print("Loaded JSON data:", data)

        # Send the data via webhook
        call_webhook(data)

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))
    except Exception as e:
        print("An error occurred:", str(e))

def call_webhook(data):
    """Send the processed data to the webhook."""
    try:
        print("Sending data to webhook...")
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 200:
            print("Webhook sent successfully:", response.status_code, response.text)
        else:
            print("Webhook response error:", response.status_code, response.text)
    except requests.RequestException as e:
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
