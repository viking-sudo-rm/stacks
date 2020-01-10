from typing import List
import argparse
import torch
import numpy as np

from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

from src.data.ud_arc_standard import UdArcStandardReader
from src.models.transition_parser import TransitionParser
from src.modules.controllers.feedforward import FeedForwardController


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

    results = model.forward_on_instances(instances)

    instance = instances[0]
    result = results[0]
    policies = result["policies"]
    pred_actions = np.argmax(policies, axis=-1)
    actions = instance["actions"]
    import pdb; pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
