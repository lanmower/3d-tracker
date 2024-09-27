import os
import subprocess
import json
import requests
import uuid
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import urllib.request

app = Flask(__name__)

# Define output directories
PRED_RESULTS_DIR = "vis_results"

# Set up your webhook URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec")

# Global ThreadPoolExecutor for managing inference tasks
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    try:
        # Run the command and capture output and error
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)  # Set timeout to 300 seconds
        print("Inference Output:", result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print("Error occurred during inference:", e.stderr)
    except subprocess.TimeoutExpired:
        print("Inference timed out.")
        return None

def download_file(image_url, output_dir, new_filename):
    """Download the image file and rename it to a new filename."""
    try:
        response = urllib.request.urlopen(image_url)
        with open(os.path.join(output_dir, new_filename), 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded and saved file as: {new_filename}")
        return os.path.join(output_dir, new_filename)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def list_files_in_directory(directory):
    """List all files in the given directory."""
    try:
        files = os.listdir(directory)
        print("Files in directory:", files)
    except Exception as e:
        print(f"Error listing files in directory: {e}")

def process_image(image_url):
    try:
        random_filename = f"{uuid.uuid4()}.mp4"  # Assuming the input is a video file
        output_dir = PRED_RESULTS_DIR

        os.makedirs(output_dir, exist_ok=True)

        downloaded_file_path = download_file(image_url, output_dir, random_filename)
        if downloaded_file_path is None:
            return  # Exit if download failed

        command = [
                "python", "body3d_img2pose_demo.py",
                "rtmdet_m_640-8xb32_coco-person.py",
                "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
                "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
                "rtmw3d-l_cock14-0d4ad840_20240422.pth", "--disable-norm-pose-2d", "--save-predictions", "--input", image_url,
                "--output-root", POSE3D_OUTPUT_DIR,
            ]

        # Run inference command
        run_inference(command)

        # List all files in the output directory after inference
        list_files_in_directory(output_dir)

        json_file_name = f"results_{random_filename.split('.')[0]}.json"
        json_file_path = os.path.join(output_dir, json_file_name)

        if os.path.exists(json_file_path):
            print(f"JSON prediction file found: {json_file_path}")
            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)
                call_webhook(json_data)
        else:
            print("Expected JSON file not found:", json_file_path)

    except Exception as e:
        print("An unexpected error occurred: ", str(e))

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
