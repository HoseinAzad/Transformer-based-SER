import torch
from torch import nn
from transformers import AutoModel


class SpeechCLF(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(SpeechCLF, self).__init__()

        self.transformer_model = AutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
        self.projector = nn.Linear(self.transformer_model.config.hidden_size, self.transformer_model.config.hidden_size)
        self.classifier = nn.Linear(self.transformer_model.config.hidden_size, num_labels)

    def freeze_feature_encoder(self):
        self.transformer_model.feature_extractor._freeze_parameters()

    def merge(self, hidden_states):
        return torch.mean(hidden_states, dim=1)

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.transformer_model(input_values, attention_mask=attention_mask).last_hidden_state
        x = self.merge(hidden_states)
        x = self.projector(x)
        x = torch.tanh(x)
        return self.classifier(x)


def get_model(check_point, num_classes, device):
    model = SpeechCLF(check_point, num_classes)
    model.freeze_feature_encoder()
    model = model.to(device)
    return model
