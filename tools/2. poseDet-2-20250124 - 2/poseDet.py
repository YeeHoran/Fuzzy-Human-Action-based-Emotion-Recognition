from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import mmcv
import numpy as np
import cv2
import os
import pickle

register_all_modules()


def load_bboxes(bbox_file):
    # Load bounding box coordinates from pickle file
    with open(bbox_file, 'rb') as f:
        bounding_boxes = pickle.load(f)
    return bounding_boxes


def main():
    datainputdir = 'data/input/AFEW'
    dataoutputKeypointFramesdir = 'data/output/AFEW' + '_keypoints'
    keypointListPickle = dataoutputKeypointFramesdir + '_keypointList.pkl'

    # Create keypoint coordinates Frame images saving dir.
    if not os.path.exists(dataoutputKeypointFramesdir):
        os.makedirs(dataoutputKeypointFramesdir)
        print('Created directory: ' + dataoutputKeypointFramesdir)
    else:
        print(dataoutputKeypointFramesdir + ' already exists')

    config_file = 'model/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'model/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'

    # Load bounding boxes from pickle file (assuming it's a dictionary)
    bbox_file = 'data/input/bbox_list.pickle'  # Path to bbox.pkl
    bounding_boxes = load_bboxes(bbox_file)

    for subdir, _, files in os.walk(datainputdir):
        # Ignore root_dir itself and only process subdirectories
        if subdir == datainputdir:
            continue
        # Split the string at '\\' and get the last part: subdir = 'data/input/path_to_output_images\\001'
        videoName_part = subdir.split('\\')[-1]

        # Collect the indices of the outer list where the sublist's string element matches
        matching_indices = [index for index, sublist in enumerate(bounding_boxes) if sublist[3] == videoName_part]
        if matching_indices:
            matching_indices=matching_indices[0]
            print(f"Found '{subdir}' at indices {matching_indices}")
        else:
            print(f"'{subdir}' not found in the list.")

        # Create a subfolder in the output directory for each subdirectory (video folder)
        subfolder_name = os.path.basename(subdir)  # Name of the video folder
        subfolder_output_path = os.path.join(dataoutputKeypointFramesdir, subfolder_name)
        if not os.path.exists(subfolder_output_path):
            os.makedirs(subfolder_output_path)

        list_keypoints_toappend=list()
        list_keypointsvisible_toappend=list()
        for filename in sorted(files):
            if filename.endswith('.png'):
                image = cv2.imread(os.path.join(subdir, filename))

                videofilenames = bounding_boxes[matching_indices][0]
                if filename in videofilenames:
                    # Get the bounding box for the current image
                    imagefile_index = videofilenames.index(filename)   # for indexing each image file in the video frames folder.
                    bbox = bounding_boxes[matching_indices][1][imagefile_index]  # Expected format: [upleft_x, upleft_y, rightbottom_x, rightbottom_y]
                    x1, y1, x2, y2 = bbox[:4]
                    y1, y2, x1, x2 = map(int, [y1, y2, x1, x2])

                    # Crop the image based on the bounding box
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image is not None and cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                        # Run inference on the cropped image using the bounding box
                        result = inference_topdown(model, cropped_image)

                        # Assuming result is your list of PoseDataSample
                        sample = result[0]

                        # Extract keypoints and visibility scores
                        keypoints = sample.pred_instances.keypoints  # shape: (1, N, 2)
                        keypoints_visible = sample.pred_instances.keypoints_visible  # shape: (1, N)

                        # Convert to 2D ndarray
                        keypoints_array = np.squeeze(keypoints)  # shape will be (N, 2)
                        keypoints_visible_array = np.squeeze(keypoints_visible)  # shape will be (N,)

                        # Adjust the keypoints back to the original image coordinates
                        for point in keypoints_array:
                            point[0] += x1  # Offset by x1
                            point[1] += y1  # Offset by y1

                        # Save the 17 keypoints of the frame to the list
                        '''
                        bounding_boxes[matching_indices] = list(bounding_boxes[matching_indices])
                        a=bounding_boxes[matching_indices]
                        list_keypoints=keypoints_array.tolist()
                        bounding_boxes[matching_indices].append(list_keypoints)
                        a=bounding_boxes[matching_indices]
                        # append the 17 keypoints and the visible score to the list
                        list_keypoints_visible=keypoints_visible_array.tolist()
                        bounding_boxes[matching_indices].append(list_keypoints_visible)
                        a = bounding_boxes[matching_indices]
                        '''
                        list_keypoints = keypoints_array.tolist()
                        list_keypoints_toappend.append(list_keypoints)
                        list_keypoints_visible = keypoints_visible_array.tolist()
                        list_keypointsvisible_toappend.append(list_keypoints_visible)
                        # Debug print statements
                        print(f"Keypoints array shape: {keypoints_array.shape}")  # Expected: (N, 2)
                        print(f"Keypoints visibility shape: {keypoints_visible_array.shape}")  # Expected: (N,)

                        # Iterate through the keypoints and draw them on the image
                        keypoint_number = -1
                        for point, visibleScore in zip(keypoints_array, keypoints_visible_array):
                            x, y = int(point[0]), int(point[1])  # Convert to integer coordinates
                            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle for each keypoint
                            # Draw the score next to the point
                            cv2.putText(image, f"{visibleScore:.2f}", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                        (255, 0, 0),
                                        1)
                            # Draw the keypoint number next to the point (adjust position slightly)
                            keypoint_number = keypoint_number + 1
                            cv2.putText(image, f"{keypoint_number}", (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                        (0, 0, 255), 1)

                            # Draw a rectangle (bounding box) on the image
                            # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                            # color is in BGR format (blue, green, red)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0),
                                          2)  # Green color, thickness of 2 pixels

                    else:
                        print("Error: Cropped image is empty or invalid.")
                        continue

                # Save the image with keypoints drawn on it
                path = os.path.join(subfolder_output_path, filename[0:-4] + '.jpg')
                cv2.imwrite(path, image)

        bounding_boxes[matching_indices] = list(bounding_boxes[matching_indices])
        a = bounding_boxes[matching_indices]
        bounding_boxes[matching_indices].append(list_keypoints_toappend)
        a = bounding_boxes[matching_indices]
        bounding_boxes[matching_indices].append(list_keypointsvisible_toappend)
        a = bounding_boxes[matching_indices]


    # Save the keypoints' coordinates of the frames for the video list to a pickle file
    with open(keypointListPickle, 'wb') as f:
        pickle.dump(bounding_boxes, f)


if __name__ == '__main__':
    main()
