import argparse
import json
import numpy as np
import os
import re
from math import ceil
import ast

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from object_detection.detection_evaluator import *

from sklearn.metrics import precision_score, recall_score
import sklearn.metrics

from evaluate import load

def calculate_f1_score(precision, recall):
    return 2 * np.multiply(precision, recall) / (precision + recall)


negative_words_reply = ["visible evidence", "sorry", "not possible", "indication", "enough information", "difficult", "hard", "impossible", "enough", "not visible", "no visible", "no violations"]
negative_words = ["visible evidence", "sorry", "not possible", "no need", "not a violation", "indication", "enough information", "difficult", "impossible", "enough", "no violations"]


def contains_keywords(s):
    # Define keyword lists for each rule
    keywords_rule0 = negative_words
    keywords_rule1 = ["PPE"]
    keywords_rule2 = ["harness"]
    keywords_rule3 = ["edge"]
    keywords_rule4 = ["radius", "excavator", "blind", "in operation"]

    # Check if any keyword from the lists is in the string, and return the rule number if found
    if any(keyword in s.lower() for keyword in keywords_rule0):
        return "0"
    elif any(keyword in s.lower() for keyword in keywords_rule1):
        return "1"
    elif any(keyword in s.lower() for keyword in keywords_rule2):
        return "2"
    elif any(keyword in s.lower() for keyword in keywords_rule3):
        return "3"
    elif any(keyword in s.lower() for keyword in keywords_rule4):
        return "4"
    else:
        return None  # No keywords found

def contains_negative_words(s, situation):

    if isinstance(s, str):
        if situation == 'reply':
            if any(keyword in s.lower() for keyword in negative_words_reply):
                return True
        elif situation == 'response':
            if any(keyword in s.lower() for keyword in negative_words):
                return True
    else:
        return False


def process_unstructured_answer(file_path):
    """
    Unstructured answers are in the form {image_id: {rule_id: explanation}}

    Input: file path

    Output:
    {
    image_id: {rule_id: {"reason": explanation}}
    }
    """

    with open(file_path, 'r') as file:
        original_dict = json.load(file)

    processed_dict = {}
    for image_id, reply in original_dict.items():
        print(image_id)
        reply = ast.literal_eval(reply)
        processed_reply = {}

        for rule_id, response in reply.items():
            processed_response = {}

            processed_response["reason"] = response
            processed_reply[rule_id] = processed_response

        processed_dict[image_id] = processed_reply

    return processed_dict


def concatenate_cot_candidates(*args):
    """
    Concatenate CoT individual predictions into one prediction with the format of either

    Input: {"rule1": file_path, "rule2": file_path, ...}

    Output:     {"0": "No violations"}, or
    {
    "1": {"reason": ..., "bounding_box": ...},
    "2": {"reason": ..., "bounding_box": ...},
    "3": {"reason": ..., "bounding_box": ...},
    "4": {"reason": ..., "bounding_box": ...}
    }
    """

    # Load
    all_candidates = {}
    i = 0
    for file_path in args:
        i += 1
        with open(file_path, 'r') as file:
            all_candidates[f"{i}"] = json.load(file)
    
    image_ids = list(all_candidates['1'].keys())
    image_ids = sorted(image_ids)

    # Concatenate
    concatenated_dict = {}
    pattern = r'\[\d+\.\d+, \d+\.\d+, \d+\.\d+, \d+\.\d+\]'

    for image_id in image_ids:
        print(image_id)
        reply = {}

        for rule_id, candidate in all_candidates.items():
            # Reasoning and bounding boxes
            original_response = candidate[image_id]
            response = {}

            # Gate1: When reply is :"{"no": ...}"" or :"{"yes": {"reason":... , "bounding_box": ...}}""
            try:
                # If it is a dictionary enclosed in a string
                original_response_dict = ast.literal_eval(original_response)
                if "no" in original_response_dict.keys():
                    # Drop the rule
                    continue
                else:
                    _, original_response = next(iter(original_response_dict.items())) # response is extracted and pass to Gate2
            except:
                pass # pass when reply is pure text, to Gate3
            
            # Gate2: When reply is :"{"reason":..., "bounding_box": ...}"
            if isinstance(original_response, dict):
                response["reason"] = original_response["reason"]
                try:
                    response["bounding_box"] = original_response["bounding box"]
                except:
                    pass

                reply[str(rule_id)] = response
                continue
            else:
                pass # pass when reply is pure text, to Gate3

            # Gate3: When reply is : "..." (pure text)
            if "No" in original_response:
                # Drop the rule
                continue
            else:
                # Case may happen that reply is : "... {}..."
                # Not a big issue for now, ok to correct them after finding correctly predicted candidates
                response["reason"] = original_response

                if re.search(pattern, response["reason"]): # extract coordinates in the response if exists
                    matches = re.findall(pattern, response["reason"])
                    coordinate_of_violation = [float(num) for num in re.findall(r'\d+\.\d+', matches[0])]
                    response["bounding_box"] = coordinate_of_violation

                reply[str(rule_id)] = response

        # If empty, reply no violations
        if len(reply) == 0:
            concatenated_dict[image_id] = {"0": "No violations"}
        else:
            concatenated_dict[image_id] = reply

        print(concatenated_dict[image_id])

    # Return
    return concatenated_dict


class VQAEvaluator:

##### ##### ##### #####  Answer preprocesses  ##### ##### ##### #####
    
    def answer_preprocess(self, answer_file):
        """The function trims the output and return the answer in required format for evaluation"""

        with open(answer_file, 'r') as file:
            answer = json.load(file)

        # Trim the answer
        modified_answer = {}
        for image_id, reply in answer.items():
            print(image_id)
            modified_reply = {}
            # print(reply)
            # Check if there is no violation
            try:
                reply = ast.literal_eval(reply)
            except:
                try:
                    if reply == "0" or "0" in reply.keys():
                        modified_answer[image_id] = {"0": "No violations"}
                        print(modified_answer[image_id])
                        continue
                except:
                    if reply == "0" or contains_negative_words(reply, situation='reply'):
                        modified_answer[image_id] = {"0": "No violations"}
                        print(modified_answer[image_id])
                        continue
            
            if isinstance(reply, set):
                modified_answer[image_id] = {"0": "No violations"}
                print(modified_answer[image_id])
                continue
            
            if "0" in reply:
                modified_answer[image_id] = {"0": "No violations"}
                print(modified_answer[image_id])
                continue

            for rule_id, response in reply.items():

                # Response may not be dict:
                try:
                    response = ast.literal_eval(response)
                except:
                    if isinstance(response, dict):
                        pass
                    else:
                        continue

                modified_response = {}
                for answer_type, explanation in response.items():
                    if answer_type.lower() == 'reason':
                        modified_explanation = explanation.strip().replace("\n", "").replace("\t", "")
                        # Remove negative reasoning
                        if contains_negative_words(modified_explanation, situation='response'):
                            break
                        else:
                            modified_response["reason"]= modified_explanation

                    if "box" in answer_type.lower():
                        modified_response["bounding_box"]= explanation
                modified_rule_id = rule_id.strip().replace("\n", "").replace("\t", "").replace("{", "")

                # Check if the modified key is a valid integer
                try:
                    int(modified_rule_id)
                    valid_rule_id = True
                except ValueError:
                    valid_rule_id = False

                # If the modified key is not a valid integer, try to find a number in the value
                if not valid_rule_id:
                    match = re.search(r'\d+', modified_explanation)
                    if match:
                        modified_rule_id = str(int(match.group()))
                        valid_rule_id = True
                    else:
                        modified_rule_id = contains_keywords(modified_explanation)
                        if modified_rule_id: 
                            valid_rule_id = True

                # Only add the key-value pair if the key is valid (an integer or has a number in the value)
                if valid_rule_id:
                    if len(modified_response) > 0:
                        modified_reply[modified_rule_id] = modified_response

            # Check if the reply is empty
            if len(modified_reply) == 0:
                modified_answer[image_id] = {"0": "No violations"}
            elif "0" in modified_reply:
                modified_answer[image_id] = {"0": "No violations"}
            else:
                # Add the modified reply to the modified answer
                modified_answer[image_id] = modified_reply

            print(modified_answer[image_id])

        # Sort the answer
        modified_answer_sorted = dict(sorted(modified_answer.items(), key=lambda item: int(item[0])))
        return modified_answer_sorted
    

##### ##### ##### #####  Multiple label classification  ##### ##### ##### #####

    def multilabel_classification_score(self, candidate_file, reference_file, isPrint=True):

        """
        Calculate the result of multi label classification. 
        Print the result.
        Return the y_pred and y_true matrices and sorted image_id list
        """

        candidates= self.answer_preprocess(candidate_file)
        with open(reference_file, 'r') as file:
            references = json.load(file)

        image_id = [key for key in references]
        sorted_image_id = sorted(image_id)
        pred_list = [[int(key) for key in candidates[img]] for img in sorted_image_id]
        label_list = [[int(key) for key in references[img]] for img in sorted_image_id]
        # print(pred_list)
        # print(label_list)


        # Convert to one-hot encoding
        y_pred = np.zeros((len(pred_list), 5))
        for i, labels in enumerate(pred_list):
            for label in labels:
                y_pred[i, label] = 1

        y_true = np.zeros((len(label_list), 5))
        for i, labels in enumerate(label_list):
            for label in labels:
                y_true[i, label] = 1
        # print(y_pred)
        # print(y_true)

        precisions = precision_score(y_true, y_pred, average=None)
        recalls = recall_score(y_true, y_pred, average=None)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average=None)

        # Print precision, recall, and F1-score for each label
        if isPrint:
            for label_index, (precision, recall, f1_score) in enumerate(zip(precisions, recalls, f1)):
                print(f"Label {label_index + 1}:")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1-score: {f1_score:.3f}")

        return y_pred, y_true, sorted_image_id


##### ##### ##### #####  Correctly predicted  ##### ##### ##### #####

    def correctly_predicted_images(self, candidate_file, reference_file, isGeneratedFile=False):
        """
        Check which image is correctly predicted for which rule and generate a storage file if needed
        """

        # Preprocess if need to generate
        if isGeneratedFile:
            candidates= self.answer_preprocess(candidate_file)
            with open(reference_file, 'r') as file:
                references = json.load(file)

        y_pred, y_true, sorted_image_id = self.multilabel_classification_score(candidate_file, reference_file, isPrint=False)
        print(y_pred)
        print(y_true)
        correctly_predicted = {
            "rule1": 1,
            "rule2": 2,
            "rule3": 3,
            "rule4": 4,
        }
    
        for rule, rule_id in correctly_predicted.items():
            matching_rows = np.where((y_pred[:, rule_id] == 1) & (y_true[:, rule_id] == 1))[0]
            matching_ids = [sorted_image_id[idx] for idx in matching_rows]
            correctly_predicted[rule] = matching_ids
        
            if isGeneratedFile:
                # Save the candidate and reference annotation in format:
                # {image_id: {'candidate': ..., 'reference': ...}}
                correctly_predicted_dict = {}
                for i in range(len(matching_ids)):
                    image_id = matching_ids[i]
                    candidate = candidates[image_id][str(rule_id)]['reason']
                    reference = references[image_id][str(rule_id)]
                    correctly_predicted_dict[image_id] = {'candidate': candidate, 'reference': reference}

                with open(rule+'_correctly_predicted.json', 'w') as file:
                    json.dump(correctly_predicted_dict, file)

        return correctly_predicted
    

    def bounding_box_score(self, candidate_file, reference_file, isGeneratedFile=True):
        """
        Calculate the IoU of bounding boxes for safety rule violation
        """

        # Preprocess if need to generate
        if isGeneratedFile:
            candidates= self.answer_preprocess(candidate_file)
            with open(reference_file, 'r') as file:
                references = json.load(file)

        y_pred, y_true, sorted_image_id = self.multilabel_classification_score(candidate_file, reference_file, isPrint=False)
        print(y_pred)
        print(y_true)
        correctly_predicted = {
            "rule1": 1,
            "rule2": 2,
            "rule3": 3,
            "rule4": 4,
        }

        IoU_list = []
        for rule, rule_id in correctly_predicted.items():
            matching_rows = np.where((y_pred[:, rule_id] == 1) & (y_true[:, rule_id] == 1))[0]
            matching_ids = [sorted_image_id[idx] for idx in matching_rows]
            correctly_predicted[rule] = matching_ids
        
            if isGeneratedFile:
                # Save the candidate and reference annotation in format:
                # {image_id: {'candidate': ..., 'reference': ...}}
                correctly_predicted_dict = {}
                for i in range(len(matching_ids)):
                    image_id = matching_ids[i]
                    print(image_id)
                    try:
                        candidate = candidates[image_id][str(rule_id)]['bounding_box']
                    except KeyError:
                        candidate = ["None"]
                    correctly_predicted_dict[image_id] = candidate

                with open(rule+'_candidate.json', 'w') as file:
                    json.dump(correctly_predicted_dict, file)
                
            IoU_list.append(get_detection_score(rule+'_candidate.json', "../../Annotations/vqa_"+rule+"_bbox_references.json"))
        
        return IoU_list
