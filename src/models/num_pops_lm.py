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
from src.utils.policy_loss import get_expected_num_pops, get_variational_loss


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
        # =================================================
        prior_distribution: Optional[List[float]] = None,
        normalize_prior: bool = True,
        pops_weight: float = 0.,
        prior_weight: float = 0.,
    ) -> None:
        super().__init__(vocab, text_field_embedder, contextualizer, dropout, num_samples,
                         sparse_embeddings, bidirectional, initializer, regularizer)

        assert isinstance(contextualizer, StackEncoder), "Contextualizer must be StackEncoder."
        assert contextualizer.store_policies, "StackEncoder needs to store its policies."

        if prior_distribution is None:
            assert prior_weight == 0, "Must have prior_weight == 0 if prior_distribution is None."

        self.pops_weight = pops_weight
        self.prior_weight = prior_weight
        self.criterion = torch.nn.MSELoss()

        # We minimize KL divergence to the prior distribution for actions.
        prior_distribution = torch.tensor(prior_distribution, device=0)
        if prior_distribution is not None and normalize_prior:
            prior_distribution = prior_distribution / torch.sum(prior_distribution)
        self.prior_distribution = prior_distribution

    def forward(  # type: ignore
        self, source: Dict[str, torch.LongTensor], lengths: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        out_dict = super().forward(source)

        if self.pops_weight > 0:
            policies = self._contextualizer.all_policies
            mask = get_text_field_mask(source)
            exp_num_pops = get_expected_num_pops(policies, mask)
            num_pops = lengths.float() - 1.
            pops_loss = self.criterion(exp_num_pops, num_pops)
        else:
            pops_loss = 0.

        if self.prior_weight > 0:
            prior_loss = get_variational_loss(policies, self.prior_distribution)
        else:
            prior_loss = 0.

        out_dict["lm_loss"] = out_dict["loss"]
        out_dict["pops_loss"] = self.pops_weight * pops_loss
        out_dict["prior_loss"] = self.prior_weight * prior_loss
        out_dict["loss"] = out_dict["lm_loss"] + out_dict["pops_loss"] + out_dict["prior_loss"]
        return out_dict
