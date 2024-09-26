import os
import subprocess
import json
import base64
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

# Global ThreadPoolExecutor for managing inference tasks
executor = ThreadPoolExecutor(max_workers=3)

def run_inference(command):
    subprocess.run(command, check=True)

def process_image(image_url, task_id):
    # Define unique directories for this task
    base_output_dir = Path("vis_results") / task_id
    pose3d_output_dir = base_output_dir / "pose3d"
    pred_results_dir = base_output_dir / "predictions"
    converted_output_dir = Path("converted_outputs") / task_id

    # Create output directories
    pose3d_output_dir.mkdir(parents=True, exist_ok=True)
    pred_results_dir.mkdir(parents=True, exist_ok=True)
    converted_output_dir.mkdir(parents=True, exist_ok=True)

    # List to store the names of the output files
    output_files = []

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
        futures = [executor.submit(run_inference, cmd) for cmd in commands]

        # Wait for all inference tasks to complete
        wait(futures)

        # Get the list of output files generated in the output directory
        output_files = list(pose3d_output_dir.glob("*")) + list(pred_results_dir.glob("*.json"))

        # Process and convert results only for the generated files
        combined_results = combine_and_convert_results(output_files, converted_output_dir)

        # Call webhook with results
        call_webhook(combined_results)

    except subprocess.CalledProcessError as e:
        print("Error occurred during inference: " + str(e))


def convert_image(input_path, output_path):
    """Convert an image to JPEG format using FFmpeg."""
    command = ["ffmpeg", "-i", str(input_path), str(output_path)]
    subprocess.run(command, check=True)


def convert_video(input_path, output_path):
    """Convert a video to MP4 format using FFmpeg."""
    if output_path.exists():
        os.remove(output_path)
    command = ["ffmpeg", "-i", str(input_path), "-c:v", "mpeg4", "-y", str(output_path)]
    subprocess.run(command, check=True)


def combine_and_convert_results(output_files, converted_output_dir):
    """Combine results and convert found images and videos to compatible formats."""
    combined_data = {}
    base64_media = {}

    # Process only output files that were generated
    for file in output_files:
        if file.is_file():
            base_filename = file.stem
            if file.suffix.lower() in [".png", ".jpg", ".jpeg"] :
                output_path = converted_output_dir / f"{base_filename}.jpg"
                convert_image(file, output_path)
                # Base64 encode the converted image
                with open(output_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
                    base64_media[base_filename] = f"data:image/jpeg;base64,{encoded_image}"
            elif file.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
                output_path = converted_output_dir / f"{base_filename}.mp4"
                convert_video(file, output_path)
                # Base64 encode the converted video
                with open(output_path, "rb") as video_file:
                    encoded_video = base64.b64encode(video_file.read()).decode("utf-8")
                    base64_media[base_filename] = f"data:video/mp4;base64,{encoded_video}"

    combined_data["base64_media"] = base64_media
    return combined_data


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
    Path("converted_outputs").mkdir(exist_ok=True)

    app.run(host="0.0.0.0", port=8080)
