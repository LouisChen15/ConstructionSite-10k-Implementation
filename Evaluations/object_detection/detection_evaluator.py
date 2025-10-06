import json
import numpy as np
import ast
import re
import os
from PIL import Image


def has_nested_list(lst):
    """
    Check if a list contains another list (nested list).

    Parameters
    ----------
    lst : list
        The list to check.

    Returns
    -------
    bool
        True if a nested list is found, False otherwise.
    """
    for elem in lst:
        if isinstance(elem, list):
            return True
    return False


def has_string(lst):
    """
    Check if a list contains at least one string.

    Parameters
    ----------
    lst : list
        The list to check.

    Returns
    -------
    bool
        True if a string is found, False otherwise.
    """
    for elem in lst:
        if isinstance(elem, str):
            return True
    return False


def normalize_value(value, max_value):
    """Normalize a coordinate and clamp to 1.0 if needed."""
    return min(round(value / max_value, 2), 1.0)


def normalize_bbox_list(bbox, width, height):
        if not bbox:  # If empty, leave unchanged
            return bbox
        x_min, y_min, x_max, y_max = bbox
        return [
            normalize_value(x_min, width),
            normalize_value(y_min, height),
            normalize_value(x_max, width),
            normalize_value(y_max, height),
        ]


def bounding_box_processing(unprocessed_boxes, normalized_scale=True, image_width=None, image_height=None):
    """
    This function take the a list of bounding boxes. 
    Returns a binary mask of size 100 X 100 with 1 indicating area enclosed by the bounding boxes. 
    """

    if isinstance(unprocessed_boxes, str):
        try:
            unprocessed_boxes = ast.literal_eval(unprocessed_boxes)
        except:
            try:
            # Regex to find all lists of numbers
                matches = re.findall(r"\[([^\]]+)\]", unprocessed_boxes)

                # Convert each match to a list of floats
                unprocessed_boxes = [list(map(float, x.split(","))) for x in matches]
            except:
                print("There is something unexpectedly wrong with the answer format. \nSo converting it to None")
                unprocessed_boxes = ["None"]

    # Calculate and return the binary mask
    if len(unprocessed_boxes) == 0 or "None" in unprocessed_boxes:
        return np.zeros((100,100))
    else:
        binary_mask = np.zeros((100,100))
        # Convert all lists to the same format of [[]]
        if not has_nested_list(unprocessed_boxes):
            unprocessed_boxes = [unprocessed_boxes]

        for unprocessed_box in unprocessed_boxes:
            if has_string(unprocessed_box):
                # Remove the outer list brackets and split the string
                numbers_str = unprocessed_box[0].strip('[]').split(',')
                # Convert each number from string to float
                unprocessed_box = [float(num) for num in numbers_str]
            elif len(unprocessed_box) < 4:
                continue

            # Only take 4 digit 
            if len(unprocessed_box) > 4:
                unprocessed_box = unprocessed_box[:4]

            # elif len(unprocessed_box) != 4:
            #     continue
            
            if normalized_scale:
                processed_box = unprocessed_box
            else:
                processed_box = normalize_bbox_list(unprocessed_box, image_width, image_height)
            print(processed_box)

            # The bounding boxes coordinates are in the format [x_min, y_min, x_max, y_max] in normalized scale
            binary_mask[
                int(processed_box[1] * 100) : int(processed_box[3] * 100) + 1, 
                int(processed_box[0] * 100) : int(processed_box[2] * 100) + 1
            ] = 1

        return binary_mask


def calculate_IoU(candidate, reference):
    """
    Takes in two binary masks and calcuate the IoU of 1.
    Output the IoU value with two sig. fig. 
    """
    # TODO: Need to consider about when there is no such instance in the image. Check the nan

    intersection = (candidate.astype(int) & reference.astype(int)).sum()
    union = (candidate.astype(int) | reference.astype(int)).sum()

    return intersection, union 


def get_detection_score(candidate_file, reference_file, type="macro", positive_only=True, normalized_scale=True, image_folder=None):
    """
    Takes in candidate and reference file path and return the precision and recall of the detection task.
    Macro calculates IoU for each instance and take the average of all the IoU.
    Micro sums all the intersection and union and take the average of the sums. 
    """
    
    with open(candidate_file, 'r') as infile:
        raw_candidate = json.load(infile)
    with open(reference_file, 'r') as infile:
        raw_reference = json.load(infile)

    image_list = sorted(list(raw_candidate.keys()))

    if type == "macro":
        IoU_list = []

        for image_id in image_list:
            print(image_id)

            if normalized_scale:
                print("Candidate mask")
                processed_candidate_mask = bounding_box_processing(raw_candidate[image_id])
                print("Reference mask")
                processed_reference_mask = bounding_box_processing(raw_reference[image_id]["normalized_scale"])
            else:
                img_path = os.path.join(image_folder, f"{image_id}.jpg")
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                print("Candidate mask")
                processed_candidate_mask = bounding_box_processing(raw_candidate[image_id], normalized_scale=normalized_scale, image_width=width, image_height=height)
                print("Reference mask")
                processed_reference_mask = bounding_box_processing(raw_reference[image_id]["pixel_scale"], normalized_scale=normalized_scale, image_width=width, image_height=height)

            if positive_only and not np.any(processed_reference_mask == 1): # if the object does not exist in this image, it is ignored
                continue
            
            intersection, union = calculate_IoU(processed_candidate_mask, processed_reference_mask)
            print(intersection, union)
            IoU_list.append(intersection/union)
        
        print(len(IoU_list))
        return np.nanmean(IoU_list)
    
    elif type == "micro":
        intersection_list = []
        union_list = []

        for image_id in image_list:
            print(image_id)

            if normalized_scale:
                print("Candidate mask")
                processed_candidate_mask = bounding_box_processing(raw_candidate[image_id])
                print("Reference mask")
                processed_reference_mask = bounding_box_processing(raw_reference[image_id]["normalized_scale"])
            else:
                img_path = os.path.join(image_folder, f"{image_id}.jpg")
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                print("Candidate mask")
                processed_candidate_mask = bounding_box_processing(raw_candidate[image_id], normalized_scale=normalized_scale, image_width=width, image_height=height)
                print("Reference mask")
                processed_reference_mask = bounding_box_processing(raw_reference[image_id]["pixel_scale"], normalized_scale=normalized_scale, image_width=width, image_height=height)

            if positive_only and not np.any(processed_reference_mask == 1): # if the object does not exist in this image, it is ignored
                continue
            
            intersection, union = calculate_IoU(processed_candidate_mask, processed_reference_mask)
            print(intersection, union)
            intersection_list.append(intersection)
            union_list.append(union)

        print(len(intersection_list))
        return sum(intersection_list)/sum(union_list)

def process_grounding_dino(candidate_file):
    """
    Grounding DINO predicts coordinates in the format of [center_x, center_y, width, height].
    This functions converts the coordinate system into [min_x, min_y, max_x, max_y].
    It will output a new file with the converted coordinates.
    """

    with open(candidate_file, 'r') as infile:
        unprocessed = json.load(infile)
    
    processed = {}
    for image_id, predictions in unprocessed.items():
        processed_predictions = []

        if "None" in predictions:
            processed_predictions.append("None")
        else:
            # Counter hallucination
            for unprocessed_prediction in predictions:
                if len(unprocessed_prediction) != 4:
                    continue
                else:
                    processed_predictions.append([
                        unprocessed_prediction[0] - unprocessed_prediction[2]/2,
                        unprocessed_prediction[1] - unprocessed_prediction[3]/2,
                        unprocessed_prediction[0] + unprocessed_prediction[2]/2,
                        unprocessed_prediction[1] + unprocessed_prediction[3]/2
                    ])

        processed[image_id] = processed_predictions

    # Add the desired suffix to the storage file
    name, ext = os.path.splitext(candidate_file)
    new_candidate_file = f"{name}_converted{ext}"

    with open(new_candidate_file, 'w') as outfile:
        json.dump(processed, outfile)
