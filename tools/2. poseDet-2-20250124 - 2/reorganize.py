import pickle
import csv
import numpy as np
# Replace 'your_file.pkl' with the path to your pickle file
with open('data/output/AFEW_keypoints_keypointList.pkl', 'rb') as f:
    data = pickle.load(f)

# get out 0-99, totally 100 data samples
data100=data[0:100]

# read in the affect label's index csv and generate the match table.
match_table = {}

# Open the CSV file
with open('circumplex_model_affect_20250126.csv', mode='r') as file:  # Replace with your file path
    csv_reader = csv.DictReader(file)

    # Iterate through each row in the CSV
    for row in csv_reader:
        # Ensure both columns are present in the row
        if 'Affect Label' in row and "Affect Label's index" in row:
            # Map the Affect Label to its index in the match table
            match_table[row['Affect Label']] = row["Affect Label's index"]

# read in the 0-99 samples and Transform 0-99 data samples' affect label to the index in the match_table, then write all of them back  to the csv file.
# Step 1: Read the existing CSV file and update the "Affect Label" column
updated_rows = []

# Open the original CSV to read
with open('PA_Label_100_20250126.csv', mode='r') as infile:  # Replace with your file path
    csv_reader = csv.DictReader(infile)
    # Convert csv_reader to a nested list (only the values of the dictionaries)
    nested_list = [list(row.values()) for row in csv_reader]

    updated_rows=list()
    # Iterate through each row in the CSV file
    for row in nested_list:
        # Transform the 'Affect Label' to its index using match_table
        affect_label = row[1]  # Access the value of the first key

        # If the affect label exists in the match_table, update it
        if affect_label in match_table:
            affect_label_index=match_table[affect_label]

        else:
            # If no match found, leave the index field blank or set as "Unknown"
            affect_label_index = "Unknown"  # or set it to a default index
        updated_row=[row[0],row[1],affect_label_index]
        # Store the updated row for writing later
        updated_rows.append(updated_row)

# Step 2: Write the updated data back to a new CSV file
with open('PA_label100_index_20250126.csv', mode='w', newline='') as outfile:  # Specify your desired file name
    csv_writer = csv.writer(outfile)

    # Write the rows from the nested_list to the CSV
    csv_writer.writerows(updated_rows)

#put affect category labels into it.
#read out each list, add a new element "affect_label" to each list(sample)
for i in range(100):
    data100[i].append(updated_rows[i][2])
    data100[i].append(updated_rows[i][1])


# Open a file in write-binary mode and save the list
with open('data/output/AFEW100_keypoints_keypointList_affectIndex20250126.pkl', 'wb') as outfile:  # Specify your desired file name
    pickle.dump(data100, outfile)



# reorganize 'AFEW100_keypoints_keypointList_affectIndex20250126.pkl' to the same structure as 'ntu_120.pkl'
# generate 'annotations' part
with open('data/output/AFEW100_keypoints_keypointList_affectIndex20250126.pkl', 'rb') as f:
    data100 = pickle.load(f)

updated_pkl_list=list()
for i in range(100):
    dictrow={}
    dictrow['frame_dir']=data100[i][3]
    dictrow['label']=int(data100[i][8])
    dictrow['image_shape'] = data100[i][4]
    dictrow['original_shape'] = data100[i][4]
    dictrow['total_frames'] = data100[i][5]
    # Convert the list to a NumPy array
    array_keypoint = np.array(data100[i][6])

    # Add a new dimension at the beginning (axis=0)
    expanded_array_keypoint = np.expand_dims(array_keypoint, axis=0)
    dictrow['keypoint'] = expanded_array_keypoint
    # Convert the list to a NumPy array
    array_keypoint_score = np.array(data100[i][7])
    # Add a new dimension at the beginning (axis=0)
    expanded_array_keypoint_score = np.expand_dims(array_keypoint_score, axis=0)
    dictrow['keypoint_score'] = expanded_array_keypoint_score

    # revise '038' 'total_frames'
    # if dictrow['label'] == '038':
    if i is 37:
        dictrow['total_frames'] = array_keypoint.shape[0]

    updated_pkl_list.append(dictrow)

# write the updated_pkl_list to 'ntu120.pkl' structure-like pkl 'AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250126.pkl'
with open('data/output/AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250126.pkl', 'wb') as outfile:
    pickle.dump(updated_pkl_list, outfile)

#generate split part
# Define the two dictionaries
xsub_train_dict = {'xsub_train': [f"{num:03}" for num in range(1, 71)]}
xsub_val_dict = {'xsub_val': [f"{num:03}" for num in range(71, 101)]}

# Create the parent dictionary
data100_dict = {}

# Add the two dictionaries to the 'split' key
data100_dict['split'] = {**xsub_train_dict, **xsub_val_dict}

# add updated_pkl_list into dict'annotations'
data100_dict['annotations'] = updated_pkl_list

# write the updated_pkl_list to 'ntu120.pkl' structure-like pkl 'AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250126.pkl'
with open('data/output/AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250127.pkl', 'wb') as outfile:
    pickle.dump(data100_dict, outfile)