import pickle

with open('ntu120_hrnet.pkl', 'rb') as f:
    data = pickle.load(f)

with open('AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250127.pkl', 'rb') as f:
    data100 = pickle.load(f)

ann=data["annotations"]

for items in ann:
    if items["total_frames"]==items["keypoint_score"].shape[1]:
        print(items['frame_dir'])

# Check what keys or data are inside the pickle file
#print(data.keys())  # This should show the structure of the dataset