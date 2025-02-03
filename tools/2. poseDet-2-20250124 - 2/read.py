import pickle
import csv
# Replace 'your_file.pkl' with the path to your pickle file
with open('data/output/AFEW100_keypoints_keypointList_affectIndex_ntu120stru_20250127.pkl', 'rb') as f:
    data = pickle.load(f)
lists=list()
for sublist in data:
    lists.append(sublist[2])

# Define the CSV file path
csv_file_path = "nested_values.csv"
# Write to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for item in lists:
        writer.writerow([item])  # Write each item in a new row

print(f"Data written to {csv_file_path}")
