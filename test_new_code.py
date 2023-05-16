import cv2
import torch
from models import FastDVDnet
from utils import remove_dataparallel_wrapper, variable_to_cv2_image
import numpy as np
from fastdvdnet import denoise_seq_fastdvdnet
from threading import Thread


# Constants
MODEL_PATH = r"/fastdvdnet/model.pth"
MAP_LOCATION = torch.device("cpu")
VIDEO_PATH = r"/fastdvdnet/test1.mp4"
NUM_IN_FR_EXT = 5
VIDEO_OUT_PATH = r"output_video.mp4"


# Normalize The Data
def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]
    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)


def frame_sequence(frame, expand_axis=True, normalize_data=True):
    # Converts the Frame Into Pixels
    frame_seq = frame

    # Convert From BGR to RGB
    frame_seq = (cv2.cvtColor(frame_seq,cv2.COLOR_BGR2RGB)).transpose(2,0,1)

    if expand_axis:
        frame_seq = np.expand_dims(frame_seq, 0)

    # Normalize The data
    if normalize_data:
        frame_seq = normalize(frame_seq)

    return frame_seq


def video2Frames_2_seq(video):
    cap = cv2.VideoCapture(video)
    seq_list = []

    if cap.isOpened():
        WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        HEIGHT =cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        FPS = cap.get(cv2.CAP_PROP_FPS)

    while (cap.isOpened()):
        ret,frame = cap.read()

        if ret == True:
            # Now Denoising The Frames
            frame_seq = frame_sequence(frame, expand_axis=False, normalize_data=True)
            seq_list.append(frame_seq)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    stacked_seq = np.stack(seq_list, axis=0)
    cap.release()
    cv2.destroyAllWindows()
    return stacked_seq, WIDTH, HEIGHT, FPS


def frames2video(denframes, noisyframes, video_name, video_width, video_height, video_fps):
    print("Making Denoosed video From Denoised Frames")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    seq_length = noisyframes.size()[0]
    video_out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, video_fps, (int(video_width), int(video_height)))

    for index in range(seq_length):
        out_img = variable_to_cv2_image(denframes[index].unsqueeze(dim=0))
        video_out.write(out_img)

    video_out.release()
    cv2.destroyAllWindows()


def denoise_worker(seq, noise_std, temp_psz, model_temporal, start, end, result):
    with torch.no_grad():
        denframes = denoise_seq_fastdvdnet(seq=seq[start:end], noise_std=noise_std,
                                           temp_psz=temp_psz, model_temporal=model_temporal)
    result[start:end] = denframes


# Main Code For Denoising
if __name__ == '__main__':
    device = torch.device("cpu")

    # Loading The Model
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    model.load_state_dict(remove_dataparallel_wrapper(torch.load(MODEL_PATH, map_location=MAP_LOCATION)))
    model.to(device)
    model.eval()

    # Converting Video to Frames and stacking into sequence
    noisyframes, WIDTH, HEIGHT, FPS = video2Frames_2_seq(video=VIDEO_PATH)

    # Setting the Parameters
    noise_std = 0.08  # Set between [0, 1]
    temp_psz = 3

    # Preparing Output array for denoised frames
    denframes = torch.zeros_like(noisyframes)

    # Create threads for denoising sequence
    num_threads = 8
    chunk_size = int(noisyframes.shape[0] / num_threads)
    threads = []
    results = [None] * noisyframes.shape[0]

    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size
        if i == num_threads - 1:
            end = noisyframes.shape[0]
        thread = Thread(target=denoise_worker, args=(noisyframes, noise_std, temp_psz, model, start, end, results))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Fill in denframes with the results
    for i in range(noisyframes.shape[0]):
        denframes[i] = results[i]

    # Saving The Denoised Video
    frames2video(denframes, noisyframes, VIDEO_OUT_PATH, WIDTH, HEIGHT, FPS)
