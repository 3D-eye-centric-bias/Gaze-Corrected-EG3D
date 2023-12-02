import numpy as np
import json
import argparse

def compute_l2_loss(json_file1, json_file2):
    with open(json_file1, 'r') as f1, open(json_file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    l2_losses = []
    
    # Create a dictionary for faster lookup from data2
    data2_dict = {item[0]: item[1] for item in data2['labels']}

    for label1 in data1['labels']:
        image_name = label1[0]
        if image_name in data2_dict:
            array1 = np.array(label1[1])
            array2 = np.array(data2_dict[image_name])
            squared_diffs = (array1 - array2) ** 2
            mse = np.mean(squared_diffs)
            l2_losses.append(mse)

    avg_l2_loss = np.mean(l2_losses)
    
    return avg_l2_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate L2 Loss (MSE) between two JSON files based on common image names.')
    parser.add_argument('--file1', type=str, required=True, help='Path to the first JSON file.')
    parser.add_argument('--file2', type=str, required=True, help='Path to the second JSON file.')
    
    args = parser.parse_args()
    
    avg_l2_loss= compute_l2_loss(args.file1, args.file2)
    print(f"Pose Score: {avg_l2_loss}")
