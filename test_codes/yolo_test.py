# from ultralytics import YOLO
# import cv2

# # Load your model
# model = YOLO("weights/helmet_model.pt")

# # Run prediction and show results
# results = model.predict("test_yolo.png", conf=0.25, show=True)

# # Show image manually
# annotated_frame = results[0].plot()
# cv2.imshow("Prediction", annotated_frame)
# cv2.waitKey(0)  # Wait for any key press
# cv2.destroyAllWindows()

from ultralytics import YOLO
model = YOLO("weights/helmet_model.pt")
print(model.names)