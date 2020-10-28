import json

import torch
import torch.nn.functional as F
import transformers
from torch import nn

from .article_classifier import ArticleClassifier

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config["SCIBERT_MODEL"])
        classifier = ArticleClassifier()
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=config["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True, 
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        confidence = {}
        selected_items = []
        
        new_keys = {'qfin': 'Quantitative Finance', 'qbio':'Quantitative Biology', 'stat': 'Statistics', 'math': 'Mathematics', 'phy': 'Physics', 'cs': 'Computer Science'}
        output = dict((new_keys[key], value) for (key, value) in output.items())
        
        for item,value in output.items():
            confidence[item] , output[item] = torch.max(F.softmax(output[item], dim=1), dim=1)
            output[item] = int(output[item])
            confidence[item] = round(float(confidence[item])*100,2)

            if(output[item] != 1):
                selected_items.append(item)
        for item in selected_items:
            output.pop(item)
            confidence.pop(item)

        return (
            confidence
        )   


model = Model()


def get_model():
    return model
