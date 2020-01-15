from typing import List
import argparse
import torch
import numpy as np

from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

from src.data.agreement import AgreementReader
from src.data.dyck import DyckReader
from src.data.eval import EvalReader
from src.data.simple_lm import SimpleLmReader
from src.decode.decoders import beam_decode, greedy_decode
from src.decode.states import DecoderState, MergeDecoderState, PushPopDecoderState, MultipopDecoderState
from src.models.num_pops_lm import NumPopsLanguageModel
from src.modules.controllers.feedforward import FeedForwardController
from src.modules.controllers.suzgun import SuzgunRnnController, SuzgunRnnCellController
from src.modules.merge_encoder import MergeEncoder
from src.modules.stack_encoder import StackEncoder
from src.utils.listener import get_policies
from src.utils.trees import from_left_distances, to_syntax_tree
from src.utils.minimalist import from_merges
from src.utils.dyck import from_parentheses


_MERGE = "merge"


def get_parse(tokens, actions, decoder_type):
    if decoder_type == "push-pop":
        return from_parentheses(tokens, actions)
    elif decoder_type == "multipop":
        return from_left_distances(tokens, actions)
    elif decoder_type == "merge":
        return from_merges(tokens, actions)
    else:
        raise ValueError("Unsupported decoder type.")


def main(args):
    vocab = Vocabulary.from_files("%s/vocabulary" % args.model_path)
    params = Params.from_file("%s/config.json" % args.model_path)

    model = Model.from_params(params=params.pop("model"), vocab=vocab)
    with open("%s/best.th" % args.model_path, "rb") as fh:
        model.load_state_dict(torch.load(fh))
    model = model.cuda(0)

    reader = DatasetReader.from_params(params.pop("dataset_reader"))
    valid_file = params.pop("validation_data_path")

    valid = reader.read(valid_file)
    instances = list(iter(valid))[:20]

    all_policies = get_policies(model, instances, "all_policies")
    all_tokens = [[tok.text for tok in instance[args.tokens_name]] for instance in instances]

    decoder_type = DecoderState.by_name(args.decoder)

    for tokens, policies in zip(all_tokens, all_policies):
        num_tokens = len(tokens)

        if args.decoder == _MERGE:
            # This should be okay here, since our neural network controller is constrained to
            # running this many times?
            policies = policies[:2 * num_tokens - 1]
        else:
            policies = policies[:num_tokens]

        if args.beam is None:
            actions = greedy_decode(policies, num_tokens, decoder_type, enforce_full=args.full)
        else:
            actions = beam_decode(policies, num_tokens, decoder_type,
                                  enforce_full=args.full,
                                  top_k=args.top_k,
                                  beam_size=args.beam)

        if actions == None:
            print("Skipping null parse.")
            continue

        pairs = set(zip(tokens, actions))
        parse, stack = get_parse(tokens, actions, args.decoder)
        output = to_syntax_tree(parse)
        import pdb; pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--tokens_name", type=str, default="source")
    parser.add_argument("--decoder", type=str, default="multipop")
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
