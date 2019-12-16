from typing import List
import argparse
import torch
import numpy as np

from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

from src.data.dyck import DyckReader
from src.decode.decoders import greedy_decode
from src.modules.stack_encoder import StackEncoder
from src.utils.listener import get_policies
from src.utils.trees import from_left_distances, to_syntax_tree
from src.utils.dyck import from_parentheses


def main(model_path):
    vocab = Vocabulary.from_files("%s/vocabulary" % model_path)
    params = Params.from_file("%s/config.json" % model_path)

    model = Model.from_params(params=params.pop("model"), vocab=vocab)
    with open("%s/best.th" % model_path, "rb") as fh:
        model.load_state_dict(torch.load(fh))
    model = model.cuda(0)

    reader = DatasetReader.from_params(params.pop("dataset_reader"))
    valid_file = params.pop("validation_data_path")

    valid = reader.read(valid_file)
    instances = list(iter(valid))[:20]

    all_policies = get_policies(model, instances, "all_policies")
    all_tokens = [[tok.text for tok in instance["source"]] for instance in instances]
    for tokens, policies in zip(all_tokens, all_policies):
        policies = policies[:len(tokens)]
        actions = greedy_decode(policies, "dyck")
        pairs = set(zip(tokens, actions))
        parse = from_parentheses(tokens, actions)
        output = to_syntax_tree(parse)
        import pdb; pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model_path)