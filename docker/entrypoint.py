import os
import subprocess
import json
import requests
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import random
import string
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

POSE3D_OUTPUT_DIR = "vis_results"
PRED_RESULTS_DIR = "vis_results"

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://nxmqazntnbzpkhmzwmue.supabase.co/functions/v1/image")
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    logger.debug(f"Running inference with command: {' '.join(command)}")
    subprocess.run(command, check=True)

def random_filename(extension, length=10):
    """ Generate a random file name with the given extension. """
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters) for _ in range(length))
    filename = f"{random_string}.{extension}"
    logger.debug(f"Generated random filename: {filename}")
    return filename

def process_image(image_url):
    os.makedirs(POSE3D_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_RESULTS_DIR, exist_ok=True)
    logger.debug(f"Created output directories: {POSE3D_OUTPUT_DIR}, {PRED_RESULTS_DIR}")

    try:
        logger.debug(f"Fetching image from URL: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        
        file_extension = image_url.split('.')[-1] if '.' in image_url else 'jpg'
        local_filename = random_filename(file_extension)
        logger.debug(f"Saving downloaded image to: {local_filename}")

        with open(local_filename, 'wb') as f:
            f.write(response.content)

        command = [
            "python", "body3d_img2pose_demo.py",
            "rtmdet_m_640-8xb32_coco-person.py",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
            "rtmw3d-l_cock14-0d4ad840_20240422.pth", 
            "--disable-rebase-keypoint", 
            "--disable-norm-pose-2d", 
            "--save-predictions", 
            "--input", local_filename,
            "--output-root", POSE3D_OUTPUT_DIR,
        ]
        
        run_inference(command)

        output_dict = {}
        output_files = [f for f in os.listdir(POSE3D_OUTPUT_DIR) if f.startswith("results_") and f.endswith(".json")]
        logger.debug(f"Output files discovered: {output_files}")
        
        for output_file in output_files:
            output_file_path = os.path.join(POSE3D_OUTPUT_DIR, output_file)
            with open(output_file_path, 'r') as f:
                file_contents = json.load(f)
                output_dict = file_contents
                logger.debug(f"Loaded contents from {output_file}: {file_contents}")

        call_webhook(output_dict)

    except Exception as e:
        logger.error("Error occurred during image processing: %s", str(e))

def call_webhook(output_dict):
    """Send the dictionary of output files to the webhook."""
    message = output_dict
    
    logger.debug(f"Sending webhook with message: {message}")
    try:
        response = requests.post(WEBHOOK_URL, json=message)
        logger.debug("Webhook response: %s - %s", response.status_code, response.text)
    except Exception as e:
        logger.error("Error calling webhook: %s", str(e))

@app.route('/process_image', methods=['GET'])
def handle_process_image():
    image_url = request.args.get('image_url')
    logger.debug(f"Received request to process image URL: {image_url}")
    
    if not image_url:
        logger.warning("image_url parameter is required")
        return jsonify({"error": "image_url parameter is required"}), 400
    
    executor.submit(process_image, image_url)
    logger.debug("Image processing started for URL: %s", image_url)
    return jsonify({"message": "Image processing started"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
