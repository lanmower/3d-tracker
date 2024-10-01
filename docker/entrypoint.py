import os
import subprocess
import json
import requests
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, wait
import uuid   # Import uuid for generating unique filenames

app = Flask(__name__)

POSE3D_OUTPUT_DIR = "vis_results"
PRED_RESULTS_DIR = "vis_results"

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec")
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    subprocess.run(command, check=True)

def download_image(image_url, filename):
    """Download the image from the given URL and save it with the provided filename."""
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download image, status code: {response.status_code}")

def process_image(image_url):
    os.makedirs(POSE3D_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_RESULTS_DIR, exist_ok=True)

    # Generate a unique filename for the image
    random_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(PRED_RESULTS_DIR, random_filename)

    try:
        # Download the image
        download_image(image_url, image_path)

        # Prepare the command to run inference using the downloaded image
        commands = [
            [
                "python", "body3d_img2pose_demo.py",
                "rtmdet_m_640-8xb32_coco-person.py",
                "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
                "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
                "rtmw3d-l_cock14-0d4ad840_20240422.pth", "--disable-rebase-keypoint", "--disable-norm-pose-2d", "--save-predictions", "--input", image_path,
                "--output-root", POSE3D_OUTPUT_DIR,
            ]
        ]
        futures = [executor.submit(run_inference, cmd) for cmd in commands]
        wait(futures)
        call_webhook()

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))
    except Exception as e:
        print("Error downloading the image: " + str(e))

def call_webhook():
    """Send a simple message to the webhook."""
    message = {"status": "Image processing completed"}
    try:
        response = requests.post(WEBHOOK_URL, json=message)
        print("Webhook response:", response.status_code, response.text)
    except Exception as e:
        print("Error calling webhook:", str(e))

@app.route('/process_image', methods=['GET'])
def handle_process_image():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "image_url parameter is required"}), 400
    executor.submit(process_image, image_url)
    return jsonify({"message": "Image processing started 321"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
