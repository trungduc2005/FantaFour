from google.colab import files
uploaded = files.upload()

import os
from google.oauth2 import service_account

# THAY BẰNG TÊN FILE JSON CỦA BẠN
SERVICE_ACCOUNT_FILE = "pregnai-agent-key.json"
PROJECT_ID = "pregnai-drns"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE
)

from IPython.display import display, Javascript
from google.colab.output import eval_js
import cv2
import base64
import numpy as np
from PIL import Image

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = '📸 Chụp ảnh';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = base64.b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Chụp ảnh
image_path = take_photo()
print("Ảnh đã lưu:", image_path)


from google.cloud import vision

def analyze_image_with_vision(image_path):
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with open(image_path, "rb") as img_file:
        content = img_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("Kết quả phân tích ảnh:")
    for label in labels:
        print(f"- {label.description} (score: {label.score:.2f})")
        if any(keyword in label.description.lower() for keyword in ["skin", "arm", "swelling", "rash", "pale", "discoloration"]):
            print("Có thể liên quan đến dấu hiệu y tế!")

# Gọi hàm
analyze_image_with_vision(image_path)


from google.cloud import dialogflow_v2 as dialogflow

def get_chatbot_response(text_input, session_id="demo-session-123"):
    session_client = dialogflow.SessionsClient(credentials=credentials)
    session = session_client.session_path(PROJECT_ID, session_id)

    text_input_obj = dialogflow.TextInput(text=text_input, language_code="vi")
    query_input = dialogflow.QueryInput(text=text_input_obj)

    response = session_client.detect_intent(
        request={"session": session, "query_input": query_input}
    )

    return response.query_result.fulfillment_text

while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ["exit", "thoát"]:
        break
    response = get_chatbot_response(user_input)
    print("PregnAI:", response)