from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, LanguageModel
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from src.modules.stack_encoder import StackEncoder
from src.utils.policy_loss import get_expected_num_pops


@Model.register("num-pops-lm")
class NumPopsLanguageModel(LanguageModel):
    """
    A LanguageModel where the number of pops is regularized to be close to n - 1.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        contextualizer: Seq2SeqEncoder,
        dropout: float = None,
        num_samples: int = None,
        sparse_embeddings: bool = False,
        bidirectional: bool = False,
        initializer: InitializerApplicator = None,
        regularizer: Optional[RegularizerApplicator] = None,
        pops_weight: float = 1.,
    ) -> None:
        super().__init__(vocab, text_field_embedder, contextualizer, dropout, num_samples,
                         sparse_embeddings, bidirectional, initializer, regularizer)

        assert isinstance(contextualizer, StackEncoder), "Contextualizer must be StackEncoder."
        assert contextualizer.store_policies, "StackEncoder needs to store its policies."

        self.criterion = torch.nn.MSELoss()
        self.pops_weight = pops_weight

    def forward(  # type: ignore
        self, source: Dict[str, torch.LongTensor], lengths: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(source)

        policies = self._contextualizer.all_policies
        mask = get_text_field_mask(source)
        exp_num_pops = get_expected_num_pops(policies, mask)
        num_pops = lengths.float() - 1.
        pops_loss = self.criterion(exp_num_pops, num_pops)
        
        output_dict["lm_loss"] = output_dict["loss"]
        output_dict["pops_loss"] = self.pops_weight * pops_loss
        output_dict["loss"] = output_dict["lm_loss"] + output_dict["pops_loss"]
        return output_dict
