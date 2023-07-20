from torch import nn
import torch
from transformers import AutoModel


class ModelForClf(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(ModelForClf, self).__init__()

        self.hubert = AutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
        self.projector = nn.Linear(self.hubert.config.hidden_size, 256)
        self.classifier = nn.Linear(256, num_labels)
        self.recurent_layer = nn.GRU(self.hubert.config.hidden_size, self.hubert.config.hidden_size  )
        self.drop = nn.Dropout(p=0.15)

    def freeze_feature_encoder(self):
        self.hubert.feature_extractor._freeze_parameters()

    def merge(self, hidden_states):
        return torch.mean(hidden_states, dim=1)

    def forward(self, input_values, attention_mask=None):
        x = self.hubert(input_values, attention_mask=attention_mask).last_hidden_state
        # x = self.transformer_model(input_values, attention_mask=attention_mask , output_hidden_states =True).hidden_states[12]

        # run the code below to : create a single vector as the representation 
        # of the entire input audio by merging the embedding vectors 
        x = self.merge(x) 

        # run the code below to :  create a single vector as the representation
        # of the entire input audio by passing all embedding vectors to a recurrent network and extracting the last hidden state 
        # x, _ = self.recurent_layer(x) 
        # x = x[:,-1] 

        x = self.projector(x)
        x = torch.tanh(x)
        logits = self.classifier(x)
        return logits



def get_model(check_point, num_classes, device):
    model = ModelForClf(check_point , num_classes)
    model.freeze_feature_encoder()
    model = model.to(device)
    return model
