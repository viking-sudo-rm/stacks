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
from src.decode.minimalist_decoders import beam_dmg_decode, greedy_dmg_decode
from src.models.expanding_classifier import ExpandingClassifier
from src.modules.compose import ComposeEncoder
from src.modules.minimalist_grammar_encoder import MinimalistGrammarEncoder
from src.modules.stack_encoder import StackEncoder
from src.utils.listener import get_policies
from src.utils.trees import from_left_distances, to_syntax_tree
from src.utils.minimalist import from_merges
from src.utils.dyck import from_parentheses


def get_parse(tokens, actions, decoder_type):
    if decoder_type == "dyck":
        return from_parentheses(tokens, actions)
    elif decoder_type == "reduce":
        return from_left_distances(tokens, actions)
    elif decoder_type == "dmg":
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
    # instances = [reader.text_to_instance(["+", "0", "1"])]

    all_policies = get_policies(model, instances, "all_policies")
    all_tokens = [[tok.text for tok in instance[args.tokens_name]] for instance in instances]
    for tokens, policies in zip(all_tokens, all_policies):
        # FIXME: Refactor this whole part. Logic is way too complicated and unorganized here.
        # There should be ONE class that handles decoding and parsing for each type.

        if args.decoder != "dmg":
            policies = policies[:len(tokens)]
        full = not args.partial

        if args.beam is None:
            actions = (greedy_dmg_decode(policies, len(tokens)) if args.decoder == "dmg" else
                       greedy_decode(policies, args.decoder, full=full))
        else:
            actions = (beam_dmg_decode(policies, len(tokens), beam_size=args.beam, top_k=args.top_k)
                       if args.decoder == "dmg" else beam_decode(policies, args.decoder,
                                                                 full=full,
                                                                 beam_size=args.beam,
                                                                 top_k=args.top_k))

        if actions == None:
            continue

        pairs = set(zip(tokens, actions))
        parse, stack = get_parse(tokens, actions, args.decoder)
        output = to_syntax_tree(parse)
        import pdb; pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--tokens_name", type=str, default="source")
    parser.add_argument("--decoder", type=str, default="dyck")
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--partial", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
