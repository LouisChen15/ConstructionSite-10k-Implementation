import argparse
import json
import numpy as np
import os
from math import ceil

from evaluate import load
from bert_score import BERTScorer

from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from coco_caption.pycocoevalcap.spice.spice import Spice
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.meteor.meteor import Meteor

from clipscore.clipscore import compute_clip_score

def split_larger_dict(dictionary, num_of_parts=20, save_folder="dict_cache"):
    """
    The function split the dictionary into 10 dictionaries and stores them into json files in the folder provided

    """

    sorted_keys = sorted(dictionary.keys())
    # Calculate the number of items per part
    items_per_part = ceil(len(sorted_keys) / num_of_parts)
    parts = []

    # Split the dictionary into parts
    for i in range(0, len(sorted_keys), items_per_part):
        part_keys = sorted_keys[i:i+items_per_part]
        part = {k: dictionary[k] for k in part_keys}
        parts.append(part)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save each part to a separate JSON file
    for i, part in enumerate(parts):
        filename = os.path.join(save_folder, f"part_{i+1}.json")
        with open(filename, "w") as file:
            json.dump(part, file)


def result_sorting(result, num_prompt:int=5, num_seed:int=5, num_data:int=613):
    """
    Sort the result according prompt number, take average of the seed
    
    Input should be 
    result:{
    "prompt1_seed1": ...
    "prompt1_seed2": ...
    ...
    "prompt2_seed1": ...
    }

    Output: numpy arrays for each prompt after taking average of 5 seeds and the original without averaging
    """

    sorted_result = np.zeros([num_prompt, num_seed, num_data])
    # Iterate through the dictionary
    for key, value in result.items():
        # Extract prompt number
        prompt = key.split("_")[-2]  
        prompt_number = int(''.join(filter(str.isdigit, prompt)))
        # Extract seed number
        seed = key.split("_")[-1]  
        seed_number = int(''.join(filter(str.isdigit, seed)))
        sorted_result[prompt_number-1, seed_number-1, :] = value
    
    # Take average
    averaged_sorted_result = np.mean(sorted_result, axis=1)
    total_averaged = np.mean(averaged_sorted_result, axis=1)

    return averaged_sorted_result, total_averaged


def calculate_f1_score(precision, recall):
    return 2 * np.multiply(precision, recall) / (precision + recall)


def consiteToCOCO(data, is_save=False):
    """ 
    This helper function aims to convert the predictions from consite dataset format to COCO dataset format for evaluation
    COCO format:
    {"image_id": [
        {"caption": ...},
        {"caption": ...},
        {"caption": ...},
        ...
    ]
    }
    """

    new_data = {k: [{'caption': v}] for k, v in data.items()}

    if is_save:
        with open("temp_consiteTOCOCO.json", "w") as file:
            json.dump(new_data, file)

    return new_data


class DescriptionEvaluator:


##### ##### ##### #####  Description preprocesses  ##### ##### ##### #####
    

    def __init__(self, need_nlp_filter:bool=False) -> None:

        self.tokenizer = PTBTokenizer()

        if need_nlp_filter:
            import stanza
            self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

    # This step makes sure labels and predictions are always at the same order
    # And store them into two lists
    def descriptions_preprocess(self, description_file:str, tokenization:bool):
        """
        Process the file and arranging the description and make sure the image id are arranged in order from min to max
        """

        descriptions = {}

        with open(description_file) as f:
            descriptions = {**descriptions, **json.load(f)}

            if tokenization:
                descriptions = consiteToCOCO(descriptions)
                descriptions = self.tokenizer.tokenize(descriptions)
                sorted_descriptions = dict(sorted(descriptions.items()))
                raw_descriptions_list = [descriptions[i][0].lower() for i in sorted_descriptions]
            else:
                sorted_descriptions = dict(sorted(descriptions.items()))
                raw_descriptions_list = [descriptions[i].lower() for i in sorted_descriptions]

        print(raw_descriptions_list[2900])
        return raw_descriptions_list
    

    def filter_descriptions(self, description_list:list, filter_level:list):
        """
        Filter the description in the file based on XPOS
        filter_level: 'list' Available choices are:
            noun: NN, NNS, NNP, NNPS
            verb: VB, VBP, VBZ, VBD, VBN, VBG
            preposition: IN
            adjective: JJ
            adverb: RB
        """

        filter_xpos = set()

        if 'noun' in filter_level:
            filter_xpos.update({'NN', 'NNS', 'NNP', 'NNPS'})
        if 'verb' in filter_level:
            filter_xpos.update({'VB', 'VBP', 'VBZ', 'VBD', 'VBN', 'VBG'})
        if 'preposition' in filter_level:
            filter_xpos.add('IN')       
            labels = [[label] for label in labels]
        if 'adjective' in filter_level:
            filter_xpos.add('JJ')
        if 'adverb' in filter_level:
            filter_xpos.add('RB')

        filtered_list = []

        for description in description_list:
            doc = self.stanza_nlp(description)
            filtered_words = [word.text for sent in doc.sentences for word in sent.words if word.xpos in filter_xpos]
            reconstructed_sentence = ' '.join(filtered_words)
            filtered_list.append(reconstructed_sentence) 
            
        return filtered_list
    

    def initialization(self, prediction_file, label_file, tokenization:bool=True, filter_level:list=None, multi_ref:bool=False):

        """
        Call other preprocess methods
        multi_ref is applicable when using aac metrics of SPICE and CIDEr
        """
        
        predictions = self.descriptions_preprocess(prediction_file, tokenization)
        if label_file:
            labels = self.descriptions_preprocess(label_file, tokenization)
        else:
            labels = None

        if filter_level:
            predictions = self.filter_descriptions(predictions, filter_level)
            if label_file:
                labels = self.filter_descriptions(labels, filter_level)


        return predictions, labels


##### ##### ##### #####  BERTScore  ##### ##### ##### #####

    def get_bert_score(self, prediction_file:str, label_file:str, filter_level:list=None, rescale_with_baseline:bool=True):
        """
        Calculate bert score and return precision, recall, and f1 score
        """
        
        predictions, labels = self.initialization(prediction_file, label_file, tokenization=False, filter_level=filter_level)
        
        bert_scorer = BERTScorer(lang='en', rescale_with_baseline=rescale_with_baseline)
        precision, recall, F1_score = bert_scorer.score(predictions, labels)
        
        return precision, recall, F1_score


##### ##### ##### #####  Bleu  ##### ##### ##### #####

    def get_bleu_score(self, prediction_file, label_file, filter_level:list=None):

        predictions, labels = self.initialization(prediction_file, label_file, filter_level)

        _bleu = load("sacrebleu")

        test_res = _bleu.compute(predictions=predictions, references=labels, lowercase=True)
        return np.array(test_res['score'])


# # The Hugging face version
# ##### ##### ##### #####  Rogue  ##### ##### ##### #####
    
#     def get_rouge_score(self, prediction_file, label_file, filter_level:list=None):
        
#         predictions, labels = self.initialization(prediction_file, label_file, filter_level)
        
#         _rouge = load("rouge")

#         test_res = _rouge.compute(predictions=predictions, references=labels)
#         # return np.mean(test_res['rouge1', 'rouge2', 'rougel'])
#         return test_res['rouge1'], test_res['rouge2'], test_res['rougeL']
    

# The COCO version
##### ##### ##### #####  Rogue  ##### ##### ##### #####
    
    def get_rouge_score(self, prediction_file, label_file, filter_level:list=None):
        """
        COCO version only returns rougeL
        """
        with open(prediction_file, 'r') as file:
            pred = json.load(file)
        with open(label_file, 'r') as file:
            lab = json.load(file)

        # convert data to COCO format
        predictions = consiteToCOCO(pred)
        labels = consiteToCOCO(lab)

        tokenizer = PTBTokenizer()
        rouge = Rouge()
        predictions  = tokenizer.tokenize(predictions)
        labels = tokenizer.tokenize(labels)

        score, scores = rouge.compute_score(labels, predictions)

        return score, scores


# The Hugging face version
##### ##### ##### #####  Meteor  ##### ##### ##### #####

    def get_meteor_score(self, prediction_file, label_file, filter_level:list=None):

        predictions, labels = self.initialization(prediction_file, label_file, filter_level)

        _meteor = load('meteor')
        
        total_score = []

        for i in range(len(predictions)):
            total_score.append(_meteor.compute(predictions=[predictions[i]], references=[labels[i]])["meteor"])

        # test_res = _meteor.compute(predictions=predictions, references=labels)
        # return np.mean(test_res['meteor']), test_res

        return sum(total_score)/len(total_score), total_score
    

# The COCO version
##### ##### ##### #####  Spice  ##### ##### ##### #####
    
    def get_spice_score(self, prediction_file, label_file, filter_level:list=None):

        """
        score returns a list of floats
        scores returns a list of dictionaries
        """

        with open(prediction_file, 'r') as file:
            pred = json.load(file)
        with open(label_file, 'r') as file:
            lab = json.load(file)

        # check file size and determines whether spliting file is needed
        pred_file_size = os.stat(prediction_file).st_size / 1024
        lab_file_size = os.stat(label_file).st_size / 1024
        toSplit = False
        if pred_file_size > 80 or lab_file_size > 80:
            toSplit = True

        # convert data to COCO format
        predictions = consiteToCOCO(pred)
        labels = consiteToCOCO(lab)

        spice = Spice()
        predictions  = self.tokenizer.tokenize(predictions)
        labels = self.tokenizer.tokenize(labels)

        if toSplit == True:
            pred_save_path = "prediction_cache"
            lab_save_path = "label_cache"
            num_of_parts = 10
            split_larger_dict(predictions, num_of_parts, pred_save_path)
            split_larger_dict(labels, num_of_parts, lab_save_path)

            score = []
            scores = []
            for i in range(num_of_parts):
                with open(os.path.join(pred_save_path, f"part_{i+1}.json"), 'r') as file:
                    predictions = json.load(file)
                with open(os.path.join(lab_save_path, f"part_{i+1}.json"), 'r') as file:
                    labels = json.load(file)
                temp_score, temp_scores = spice.compute_score(labels, predictions)
                score.append(temp_score)
                scores.append(temp_scores)
        else:
            score, scores = spice.compute_score(labels, predictions)

        return score, scores

# The COCO version
##### ##### ##### #####  Cider  ##### ##### ##### #####
    
    def get_cider_score(self, prediction_file, label_file, filter_level:list=None):

        """
        Returns 
        """

        with open(prediction_file, 'r') as file:
            pred = json.load(file)
        with open(label_file, 'r') as file:
            lab = json.load(file)

        # convert data to COCO format
        predictions = consiteToCOCO(pred)
        labels = consiteToCOCO(lab)

        cider = Cider()
        predictions  = self.tokenizer.tokenize(predictions)
        labels = self.tokenizer.tokenize(labels)

        score, scores = cider.compute_score(labels, predictions)

        return score, scores
    

##### ##### ##### #####  FAIEr  ##### ##### ##### #####
    
    def get_faier_score(self, prediction_file, label_file, filter_level:list=None):

        """
        The original code is written in Python 2, too complicated to rewrite, refer to FAIEr files for evaluation.
        """

        return "The original code is written in Python 2, too complicated to rewrite, refer to FAIEr files for evaluation."
    

##### ##### ##### #####  CLIPScore  ##### ##### ##### #####
    
    def get_clip_score(self, image_folder, prediction_file, label_file=None, filter_level:list=None):

        """
        Returns numpy arrays
        """

        predictions, labels = self.initialization(prediction_file, label_file, tokenization=False, filter_level=None, multi_ref=False)

        # assume image name same as prediction id
        with open(prediction_file, 'r') as file:
            prediction_dict = json.load(file)
            image_list = list(prediction_dict.keys())
            image_list = sorted(image_list)
        image_paths = [os.path.join(image_folder, image_id + ".jpg") for image_id in image_list]

        if label_file:
            clip_score, ref_clip_score, clip_scores, ref_clip_scores = compute_clip_score(image_paths, predictions, labels)
            return clip_score, ref_clip_score, clip_scores, ref_clip_scores
        
        else:
            clip_score, clip_scores = compute_clip_score(image_paths, predictions, labels)
            return clip_score, clip_scores

##### ##### ##### #####  Word Count  ##### ##### ##### #####
    
    def get_word_count(self, text_file):

        """
        Return average words per caption
        """

        # assume image name same as prediction id
        with open(text_file, 'r') as file:
            total_dict = json.load(file)
        
        total_words = 0
        for key, value in total_dict.items():
            total_words += len(value.split())

        average_word = total_words / len(total_dict)

        return average_word
