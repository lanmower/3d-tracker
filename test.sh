docker run --gpus all -v ./vis_results:/mmpose/vis_results -v ./pred_results:/mmpose/pred_results -p 5000:5000 -e IMAGE_URL=https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4 -e WEBHOOK_URL=https://script.google.com/macros/s/AKfycbx25RhbUQ3_Otyy1Jm1B3JDuH0jZUUAl56HObeH02mYzTPebMYj2Vy9v3tL6FW1gFwq/exec --shm-size=8g  mmpose

curl -X POST https://localhost:5000/process_image \
    -H "Content-Type: application/json" \
    -d '{"image_path": "https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4"}'

