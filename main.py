import cv2
from time import time
import torch
from models import FastDVDnet
from utils import remove_dataparallel_wrapper, variable_to_cv2_image
import numpy as np
from fastdvdnet import temp_denoise
from pprint import pprint


# Constants

MODEL_PATH = r"C:\Users\GANESH ESLAVATH\Documents\fastdvdnet-updated\fastdvdnet\model.pth"
MAP_LOCATION = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_PATH = r"C:\Users\GANESH ESLAVATH\Documents\fastdvdnet-updatedta\fastdvdnet\Test_Video2.mp4"
NUM_IN_FR_EXT = 5
VIDEO_OUT_PATH = r"output_video.mp4"

# Normalize The Data
def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def frame_sequence(frame,expand_axis = True,normalize_data= True):
	# Converts the Frame Into Pixels
	frame_seq = frame

	# Convert FRom BGR to  RGB
	frame_seq = (cv2.cvtColor(frame_seq,cv2.COLOR_BGR2RGB)).transpose(2,0,1)

	if expand_axis:
		frame_seq = np.expand_dims(frame_seq,0)
	# Normalize The data
	if normalize_data:
		frame_seq = normalize(frame_seq)

	return  frame_seq

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
	return stacked_seq,WIDTH,HEIGHT,FPS

def frames2video(denframes,noisyframes,video_name,video_width,video_height,video_fps):
	print("Making Denoosed video From Denoised Farmes")
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	seq_length = noisyframes.size()[0]
	video_out = cv2.VideoWriter(VIDEO_OUT_PATH,fourcc,video_fps,(int(video_width),int(video_height)))
	for index in range(24):

		out_img = variable_to_cv2_image(denframes[index].unsqueeze(dim=0))
		video_out.write(out_img)

	# video_out.release()
	video_out.release()
	cv2.destroyAllWindows()


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal):
	r"""Denoises a sequence of frames with FastDVDnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes.append(seq[relidx])

		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		denframes[fridx] = temp_denoise(model_temporal, inframes_t, noise_map)

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

# Main Code For Denoising

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Loading The Model
model_temp = FastDVDnet(num_input_frames=5)  #Depending on the Num Of frames Denoising Is Dome

# Loading Previous Weights

state_temp_dict = torch.load(MODEL_PATH,map_location=MAP_LOCATION)

# Due To abscence Of Gpu we are Removing data_paraller_wrapper

state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

model_temp.load_state_dict(state_temp_dict)

model_temp.eval()

with torch.no_grad():

	stacked_sequence,width,height,fps= video2Frames_2_seq(video=VIDEO_PATH)

	stacked_sequence = torch.from_numpy(stacked_sequence).to(device)

	# Noise Can Be Depending On the Frames which are Are Noisy We should estimate usimg some algorithm

	noise_sigma = 30/255.
# 	Estimating Or guessing The Noise
	noise = torch.empty_like(stacked_sequence).normal_(mean=0, std=noise_sigma).to(device)
# 	No need Of Adding Noise If necessary Feel Free To ADD aTHE adding Noise the Frames

	noisestd = torch.FloatTensor([noise_sigma]).to(device)
	# Denoised Sequence Is Formed
	t1= time()
	denframes = denoise_seq_fastdvdnet(seq=stacked_sequence,
									   noise_std=noisestd,
									   temp_psz=NUM_IN_FR_EXT,
									   model_temporal=model_temp)
	t2 = time()

# Video Writer

frames2video(denframes=denframes,noisyframes=stacked_sequence,video_name="Latest_out",video_width=width,video_height=height,video_fps=fps)

print(t1-t2)
print("Done!")















