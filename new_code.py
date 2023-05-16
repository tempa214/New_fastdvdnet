import cv2
import torch
import numpy as np
from models import FastDVDnet

# Initialize FastDVDnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastDVDnet(num_input_frames=5, num_output_frames=1)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
model.to(device)

# Capture video from camera
cap = cv2.VideoCapture(0)

# Set width and height of video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert to tensor
        tensor = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.
        tensor = tensor.to(device)

        # Denoise tensor
        with torch.no_grad():
            output = model(tensor)
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.

        # Display denoised frame
        cv2.imshow('frame', output.astype(np.uint8))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
