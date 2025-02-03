import json
import os
import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import pickle


def read_arousal_valence(json_file_path, video_id=None):
    # Open the JSON file with explicit UTF-8 encoding
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    video_id=data.get('video_id')

    # Extract 'arousal' and 'valence' values
    frames=data.get("frames")
    frame=frames.get("00000")
    arousal = frame.get('arousal')
    valence = frame.get('valence')

    return arousal, valence, video_id


def process_image(image_path):
    img = read_image(image_path)
    # Get the height and width
    _, height, width = img.shape

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()
    preprocess = weights.transforms()
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"], labels=labels, colors="red", width=4, font_size=30)
    im = to_pil_image(box.detach())
    tensor_image = transforms.ToTensor()(im)

    # Return the first detected person's box and score, or None and 0.0 if none detected
    if prediction["boxes"].nelement() > 0:
        return tensor_image, prediction['boxes'][0], prediction['scores'][0].item()
    else:
        return tensor_image, None, 0.0


if __name__ == '__main__':
    # Root directory containing subfolders (e.g., 001, 002, ..., 050)
    root_dir = 'AFEW'  # You can change this to the directory containing subfolders for video frames
    # Output directory for processed images
    output_dir = 'path_to_output_images'  # Make sure this is the root directory for output
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    list_person_positions = []
    list_filenames = []
    list_VA=[]
    list_videoID=[]
    list_videoHeightWidth=[]
    list_totalFrames=[]

    # Walk through all subdirectories and files in the root directory
    for subdir, _, files in os.walk(root_dir):
        # Ignore root_dir itself and only process subdirectories
        if subdir == root_dir:
            continue

        # Create a subfolder in the output directory for each subdirectory (video folder)
        subfolder_name = os.path.basename(subdir)  # Name of the video folder
        subfolder_output_path = os.path.join(output_dir, subfolder_name)
        if not os.path.exists(subfolder_output_path):
            os.makedirs(subfolder_output_path)

        # Initialize lists for each video folder
        video_filenames = []
        video_person_positions = []

        # Get the total number of files
        total_files = len(files)-1   #one is json file, not video frame
        list_totalFrames.append(total_files)
        # Process files in the subdirectory (video frame folder)
        for idx, filename in enumerate(sorted(files)):
            if filename.endswith('.png'):  # Or '.jpg' if needed
                input_path = os.path.join(subdir, filename)
                video_filenames.append(filename)

                processed_image, person_box_position, score= process_image(input_path)

                # Convert box position and score to list (if not None)
                if person_box_position is not None:
                    person_box_position = person_box_position.tolist()
                else:
                    person_box_position = [0, 0, 0, 0]  # Placeholder for no detection

                score = float(score)  # Ensure score is a float

                # Append position and score together (as a single list or tuple)
                # Here we combine them into a single list for simplicity
                person_data = person_box_position + [score]
                video_person_positions.append(person_data)

                # Construct output image path and save the processed image
                output_path = os.path.join(subfolder_output_path, f'{idx + 1:04d}{os.path.splitext(filename)[1]}')
                torchvision.utils.save_image(processed_image, output_path)
                print(f'Processed and saved: {output_path}')
            elif filename.endswith('.json'):
                json_path = os.path.join(subdir, filename)
                json_file_path = os.path.join(subdir, filename)
                if os.path.exists(json_file_path):
                    arousal, valence, video_ID = read_arousal_valence(json_file_path)
                    list_VA.append((valence, arousal))
                    list_videoID.append(video_ID)



        # Append each video folder's filenames, positions, videoHeightWidth to the global list
        list_filenames.append(video_filenames)
        list_person_positions.append(video_person_positions)
        list_videoHeightWidth.append((720,576))




    # Save the list of bounding boxes with filenames to a pickle file
    bounding_boxes = list(zip(list_filenames, list_person_positions, list_VA, list_videoID, list_videoHeightWidth, list_totalFrames))
    with open('bbox_list.pickle', 'wb') as f:
        pickle.dump(bounding_boxes, f)
    print("Bounding boxes saved to 'bbox_list.pickle'")

