import cv2

def get_camera_streams(camera_urls):
    caps = []
    for url in camera_urls:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Failed to open camera stream: {url}")
        caps.append(cap)
    return caps
