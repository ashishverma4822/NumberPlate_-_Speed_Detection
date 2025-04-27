import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
import time
import os

# --- Title and Sidebar ---
st.set_page_config(page_title="Automatic Vehicle Violation Detection - Lost Set Bit", layout="wide")
st.sidebar.title("Lost Set Bit 🚀")
st.sidebar.write("Vehicle Speed & Number Plate Detection System")

# --- Load Models ---
car_model_path = 'yolov8s.pt'  # Your car detection model
plate_model_path = '/content/drive/MyDrive/YOLO-car-license-plate-detection-main/temp/license_plate_detector.pt'  # Your plate detection model

try:
    car_model = YOLO('yolov8s.pt')
    plate_model = YOLO(plate_model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")

reader = easyocr.Reader(['en'])

# --- Utility Functions ---
def extract_text_from_image(cropped_img):
    result = reader.readtext(cropped_img)
    if result:
        return result[0][-2]
    return "No Plate Detected"

def calculate_speed(pos1, pos2, time1, time2, fixed_distance_meters=10):
    # Using euclidean distance (optional fixed distance)
    time_diff = time2 - time1
    if time_diff > 0:
        speed_mps = fixed_distance_meters / time_diff
        speed_kmph = speed_mps * 3.6
        return round(speed_kmph, 2)
    return 0

# --- Process Image ---
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    car_results = car_model.predict(image_rgb, device='cpu')
    license_plates_data = []

    for car_result in car_results:
        for car_box in car_result.boxes:
            x1, y1, x2, y2 = map(int, car_box.xyxy[0])
            car_crop = image_rgb[y1:y2, x1:x2]

            plate_results = plate_model.predict(car_crop, device='cpu')
            for plate_result in plate_results:
                for plate_box in plate_result.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_crop = car_crop[py1:py2, px1:px2]

                    plate_text = extract_text_from_image(plate_crop)
                    confidence = plate_box.conf[0]

                    if plate_text != "No Plate Detected" and len(plate_text) >= 6 and confidence >= 0.6:
                        license_plates_data.append({
                            "plate_text": plate_text,
                            "car_box": (x1, y1, x2, y2),
                            "plate_box": (px1 + x1, py1 + y1, px2 + x2, py2 + y2)
                        })

                        # Draw car box
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image_rgb, f'Car', (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                        # Draw plate box
                        cv2.rectangle(image_rgb, (px1 + x1, py1 + y1), (px2 + x2, py2 + y2), (0, 255, 0), 2)
                        cv2.putText(image_rgb, f'{plate_text}', (px1 + x1, py1 + y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = os.path.join("/content/drive/MyDrive/YOLO-car-license-plate-detection-main/temp", "output_image.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    return license_plates_data, output_path

# --- Process Video or Webcam ---
def process_video(video_path, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    vehicle_tracks = {}
    frame_idx = 0
    video_placeholder = st.empty()
    table_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        car_results = car_model.predict(frame_rgb, device='cpu')

        for car_result in car_results:
            for car_box in car_result.boxes:
                x1, y1, x2, y2 = map(int, car_box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                matched = False
                for plate_text in vehicle_tracks:
                    last_center = vehicle_tracks[plate_text]['last_center']
                    dist = np.linalg.norm(np.array(center) - np.array(last_center))
                    if dist < 50:  # Threshold for same car
                        vehicle_tracks[plate_text]['last_center'] = center
                        vehicle_tracks[plate_text]['last_time'] = timestamp
                        matched = True
                        break

                if not matched:
                    car_crop = frame_rgb[y1:y2, x1:x2]
                    plate_results = plate_model.predict(car_crop, device='cpu')

                    for plate_result in plate_results:
                        for plate_box in plate_result.boxes:
                            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                            plate_crop = car_crop[py1:py2, px1:px2]
                            plate_text = extract_text_from_image(plate_crop)
                            confidence = plate_box.conf[0]

                            if plate_text != "No Plate Detected" and len(plate_text) >= 6 and confidence >= 0.6:
                                vehicle_tracks[plate_text] = {
                                    'first_center': center,
                                    'last_center': center,
                                    'first_time': timestamp,
                                    'last_time': timestamp
                                }

                # Draw car box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Update Streamlit
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if vehicle_tracks:
            df = pd.DataFrame({
                "License Plate": list(vehicle_tracks.keys()),
                "First Detected Time (s)": [round(v["first_time"], 2) for v in vehicle_tracks.values()],
                "Last Detected Time (s)": [round(v["last_time"], 2) for v in vehicle_tracks.values()],
                "Estimated Speed (km/h)": [calculate_speed(v["first_center"], v["last_center"], v["first_time"], v["last_time"]) for v in vehicle_tracks.values()]
            })
            table_placeholder.dataframe(df)

        time.sleep(0.05)

    cap.release()
    return vehicle_tracks

# --- Main App Logic ---
st.title("Automatic Vehicle Violation Detection System 🚗🚦")

option = st.radio("Choose input type:", ("Upload Image", "Upload Video", "Use Webcam"), horizontal=True)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        input_path = os.path.join("/content/drive/MyDrive/YOLO-car-license-plate-detection-main/temp", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Processing image...")
        plates_data, output_img_path = process_image(input_path)

        st.image(output_img_path, caption="Processed Image", use_container_width=True)

        if plates_data:
            df = pd.DataFrame({
                "License Plate": [p['plate_text'] for p in plates_data],
                "Speed (km/h)": ["N/A"] * len(plates_data)
            })
            st.dataframe(df)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        input_path = os.path.join("/content/drive/MyDrive/YOLO-car-license-plate-detection-main/temp", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Processing video...")
        vehicle_tracks = process_video(input_path)

elif option == "Use Webcam":
    st.write("Starting webcam...")
    vehicle_tracks = process_video(None, use_webcam=True)

st.sidebar.write("---")
st.sidebar.markdown("**Built with ❤️ by Lost Set Bit**")
