import torch
from torch import nn
from transformers import AutoModel

from tkg_qa_models.data import LM_NAMES
from tkg_qa_models.kge import score_entities, score_timestamps


class QAModel(nn.Module):
    """
    Shared parts of all QA models: the language model, the pretrained TComplEx tables and the
    combined entity/time table used to look up question slots.

    Index spaces follow tkg_qa_models.data: entity ids, then num_entities + timestamp id, and
    a zero padding row at num_entities + num_timestamps.

    forward(batch, mode) returns `scores` [B, num_entities + num_timestamps] in eval mode and
    `(scores, loss)` in train mode (or `(scores, hop_weights)` for multi-hop models in eval mode).
    """

    entity_aware = False        # True: the dataset masks mentions and the model injects their embeddings

    def __init__(self, tkbc_model, args, lm=None):
        super().__init__()
        self.args = args
        self.tkbc_model = tkbc_model
        self.rank = tkbc_model.rank
        self.tkbc_embedding_dim = tkbc_model.entity_table.shape[1]
        self.num_entities = tkbc_model.num_entities
        self.num_times = tkbc_model.num_timestamps
        self.hard = getattr(args, "supervision", "soft") == "hard"

        self.lm_name = LM_NAMES.get(lm or getattr(args, "lm", "distilbert"), lm or args.lm)
        self.lm_frozen = args.lm_frozen == 1
        # a fine-tuned LM uses eager attention: MPS has no fused attention with dropout
        self.lm_model = AutoModel.from_pretrained(self.lm_name, **({} if self.lm_frozen else {"attn_implementation": "eager"}))
        self.sentence_embedding_dim = self.lm_model.config.hidden_size
        if self.lm_frozen:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        full = torch.cat([tkbc_model.entity_table.data, tkbc_model.time_table.data], dim=0)
        self.padding_idx = full.shape[0]
        self.entity_time_embedding = nn.Embedding(full.shape[0] + 1, self.tkbc_embedding_dim,
                                                  padding_idx=self.padding_idx)
        self.entity_time_embedding.weight.data[:-1].copy_(full)
        if args.frozen == 1:
            self.entity_time_embedding.weight.requires_grad = False
            for param in self.tkbc_model.parameters():
                param.requires_grad = False

        self.loss = nn.CrossEntropyLoss(reduction="mean")          # ignores answer == -100
        self.loss_dict = {}

    def train(self, mode=True):
        super().train(mode)
        if self.lm_frozen:
            self.lm_model.eval()                 # a frozen encoder runs without dropout
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def inputs(self, batch, *keys):
        return [batch[k].to(self.device) for k in keys]

    def cls_embedding(self, input_ids, attention_mask):
        return self.lm_model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def score_entities(self, lhs, rel, time):
        return score_entities(lhs, rel, time, self.tkbc_model.entity_table, self.rank)

    def score_timestamps(self, lhs, rel, rhs):
        return score_timestamps(lhs, rel, rhs, self.tkbc_model.time_table, self.rank)

    def hint_embeddings(self, batch):
        """TKG embeddings of the retrieved (t1, t2) hint (the raw-TKG `-Hard` supervision)."""
        t1, t2 = self.inputs(batch, "t1", "t2")
        return self.tkbc_model.embeddings[2](t1), self.tkbc_model.embeddings[2](t2)

    def single_answer_loss(self, scores, batch):
        loss = self.loss(scores, batch["answer"].to(self.device))
        self.loss_dict = {"total": loss.detach()}
        return loss
