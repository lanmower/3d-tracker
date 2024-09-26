import os
import subprocess
import json
import base64
import requests
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

app = Flask(__name__)

# Define output directories
POSE3D_OUTPUT_DIR = "vis_results"
PRED_RESULTS_DIR = "vis_results"
CONVERTED_OUTPUT_DIR = "converted_outputs"  # Directory for converted files

# Set up your webhook URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec")

# Global ThreadPoolExecutor for managing inference tasks
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    subprocess.run(command, check=True)

def process_image(image_url):
    # Create output directories if they do not exist
    os.makedirs(POSE3D_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_RESULTS_DIR, exist_ok=True)
    os.makedirs(CONVERTED_OUTPUT_DIR, exist_ok=True)

    try:
        # Define the inference commands
        commands = [
            [
                "python", "body3d_img2pose_demo.py",
                "rtmdet_m_640-8xb32_coco-person.py",
                "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
                "configs/rtmw3d-l_8xb64_cocktail14-384x288.py",
                "rtmw3d-l_cock14-0d4ad840_20240422.pth", "--disable-rebase-keypoint", "--disable-norm-pose-2d", "--save-predictions", "--input", image_url,
                "--output-root", POSE3D_OUTPUT_DIR,
            ]
        ]

        # Submit inference commands to executor
        futures = [executor.submit(run_inference, cmd) for cmd in commands]

        # Wait for all inference tasks to complete
        wait(futures)

        # Process and convert results
        combined_results = combine_and_convert_results()

        # Call webhook with results
        call_webhook(combined_results)

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))

def convert_image(input_path, output_path):
    """Convert an image to JPEG format using FFmpeg."""
    command = ['ffmpeg', '-i', input_path, output_path]
    subprocess.run(command, check=True)

def convert_video(input_path, output_path):
    """Convert a video to MP4 format using FFmpeg."""
    if os.path.exists(output_path):
        os.remove(output_path)
    command = ['ffmpeg', '-i', input_path, '-c:v', 'mpeg4', '-y', output_path]
    subprocess.run(command, check=True)

def combine_and_convert_results():
    """Combine results and convert found images and videos to compatible formats."""
    combined_data = {}
    base64_images = {}

    # Combine predictions
    for root, dirs, files in os.walk(PRED_RESULTS_DIR):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)
                    base_filename = os.path.splitext(file)[0]
                    combined_data[base_filename] = data

    # Convert and combine images and videos
    for root, dirs, files in os.walk(POSE3D_OUTPUT_DIR):
        for file in files:
            input_path = os.path.join(root, file)
            base_filename = os.path.splitext(file)[0]
            if file.endswith(('.png', '.jpg', '.jpeg')):  
                output_path = os.path.join(CONVERTED_OUTPUT_DIR, f"{base_filename}.jpg")
                convert_image(input_path, output_path)
                # Base64 encode the converted image
                with open(output_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                    base64_images[base_filename] = f"data:image/jpeg;base64,{encoded_image}"
            elif file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  
                output_path = os.path.join(CONVERTED_OUTPUT_DIR, f"{base_filename}.mp4")
                convert_video(input_path, output_path)
                # Base64 encode the converted video
                with open(output_path, "rb") as video_file:
                    encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
                    base64_images[base_filename] = f"data:video/mp4;base64,{encoded_video}"

    combined_data['base64_images'] = base64_images
    return combined_data

def call_webhook(data):
    """Send the processed data to the webhook."""
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        print("Webhook response:", response.status_code, response.text)
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
