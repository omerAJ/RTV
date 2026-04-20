import cv2

# Use the correct device
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

for i in range(100):  # read 100 frames
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Frame is a NumPy array
    print(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}")

cap.release()
print("Done")