import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


class OKVQAEvalData(torch.utils.data.Dataset):
    """This class is made simply meet the format requirement of the original author"""
    
    def __init__(self, vis_processor, image_path, prompt):
        self.vis_processor = vis_processor
        self.total_image_path = [image_path]
        self.prompt = prompt
    
    def __len__(self):
        return len(self.total_image_path)

    def __getitem__(self, idx):
        image_path = self.total_image_path[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        
        return image, self.prompt

class Predictor():

    def __init__(self):
        parser = eval_parser()
        args = parser.parse_args()
        self.cfg = Config(args)

        self.model, self.vis_processor = init_model(args)
        self.conv_temp = CONV_VISION_minigptv2.copy()
        self.conv_temp.system = ""
        self.model.eval()
        
    
    def predict(
        self,
        image_path,
        prompt,
        top_p=1.0,
        temperature=0.2,
        max_tokens=1024,
    ):
        """Run a single prediction on the model""" 

        data = OKVQAEvalData(self.vis_processor, image_path, prompt)
        eval_dataloader = DataLoader(data, batch_size=1, shuffle=False)

        for image, question in eval_dataloader:

            texts = prepare_texts(question, self.conv_temp)  # warp the texts with conversation template
            answer = self.model.generate(image, texts, max_new_tokens=1024, do_sample=False)

            answer = answer[0].replace('<unk>','')

        return answer
