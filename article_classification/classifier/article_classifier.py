import json

import transformers
from torch import nn

with open("config.json") as json_file:
    config = json.load(json_file)


class ArticleClassifier(nn.Module):
    def __init__(self):
        super(ArticleClassifier, self).__init__()
        self.scibert = transformers.AutoModel.from_pretrained(config["SCIBERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)

        self.cs = nn.Linear(self.scibert.config.hidden_size,2)
        self.phy = nn.Linear(self.scibert.config.hidden_size,2)        
        self.math = nn.Linear(self.scibert.config.hidden_size,2)        
        self.stat = nn.Linear(self.scibert.config.hidden_size,2)        
        self.qbio = nn.Linear(self.scibert.config.hidden_size,2)            
        self.qfin = nn.Linear(self.scibert.config.hidden_size,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.scibert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = self.drop(pooled_output)    
        
        output = {}
        output['cs'] = self.cs(pooled_output)
        output['phy'] = self.phy(pooled_output)        
        output['math'] = self.math(pooled_output)        
        output['stat'] = self.stat(pooled_output)        
        output['qbio'] = self.qbio(pooled_output)                
        output['qfin'] = self.qfin(pooled_output)     
        
        return output           
