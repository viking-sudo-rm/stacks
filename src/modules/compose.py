from overrides import overrides
from typing import List

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("compose")
class ComposeEncoder(Seq2SeqEncoder):

    """Compose several Seq2SeqEncoders to get a new one.

    This can be useful for adding a contextualizer in front of an arbitrary layer type.
    """

    def __init__(self, encoders: List[Seq2SeqEncoder]):
        super().__init__()
        self.encoders = encoders
        for idx, encoder in enumerate(encoders):
            self.add_module("encoder%d" % idx, encoder)

        # Compute bidirectionality.
        all_bidirectional = all(encoder.is_bidirectional() for encoder in encoders)
        any_bidirectional = any(encoder.is_bidirectional() for encoder in encoders)
        self.bidirectional = all_bidirectional

        if all_bidirectional != any_bidirectional:
            raise ValueError("All encoders need to match in bidirectionality.")

        if len(self.encoders) < 1:
            raise ValueError("Need at least one encoder.")

        last_encoder = None
        for encoder in encoders:
            if (last_encoder is not None and
                    last_encoder.get_output_dim() != encoder.get_input_dim()):
                raise ValueError("Encoder input and output dimensions don't match.")
            last_encoder = encoder

    @overrides
    def forward(self, inputs, mask):
        for encoder in self.encoders:
          inputs = encoder(inputs, mask)
        return inputs

    @overrides
    def get_input_dim(self) -> int:
        return self.encoders[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.encoders[-1].get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional
