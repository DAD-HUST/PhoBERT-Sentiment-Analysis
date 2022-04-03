import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel


class RobertaVN(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaVN, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.linear = nn.Linear(4 * config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        # concat 4 first outputs
        cls_output = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...],
                               outputs[2][-3][:, 0, ...], outputs[2][-4][:, 0, ...]), -1)
        logits = self.linear(cls_output)
        return logits
