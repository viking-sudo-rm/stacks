import os
from token import NAME, INDENT, DEDENT
from tokenize import tokenize, untokenize
from typing import List
from listify import listify
import random

DELIM = "\n\n"

REPO_NAME = "tensorflow"
REPO_PATH = os.path.join("/home/willm/code", REPO_NAME)
OUT_PATH = os.path.join("/net/nfs.corp/allennlp/willm/data", REPO_NAME)
MODEL_PATH = os.path.join("/tmp/willm/stacks/lm", REPO_NAME)


def _reset_state():
    """Reset the tokenizer state for extracting function bodies."""
    return -1, []


@listify
def get_function_blocks(repo_path: str):
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith(".py"):
                path = os.path.join(dirpath, filename)
                with open(path, "rb") as fh:
                    depth, tokens = _reset_state()
                    for tok_type, tok_string, _, _, _ in tokenize(fh.readline):

                        # Logic for when we are not in any function body.
                        if depth == -1:
                            if tok_type == NAME and tok_string == "def":
                                depth = 0
                            else:
                                continue

                        token = (tok_type, tok_string)
                        tokens.append(token)

                        if tok_type == INDENT:
                            depth += 1
                        elif tok_type == DEDENT:
                            depth -= 1
                            if depth == 0:
                                yield tokens
                                depth, tokens = _reset_state()


def split_train_test_valid(blocks: List[str]):
    random.shuffle(blocks)
    end_train = int(len(blocks) * .8)
    end_test = int(len(blocks) * .9)

    train = blocks[:end_train]
    test = blocks[end_train:end_test]
    valid = blocks[end_test:]
    return train, test, valid


def save_code_blocks(blocks: List[str], dir_path: str):
    os.makedirs(dir_path)
    for idx, block in enumerate(blocks):
        path = os.path.join(dir_path, "block%d.py" % idx)
        with open(path, "w") as fh:
            fh.write(block)


def main():
    print("Finding function blocks in %s..." % REPO_PATH)
    blocks = get_function_blocks(REPO_PATH)
    print("Untokenizing function blocks...")
    blocks = [untokenize(block) for block in blocks]
    print("Splitting train, test, valid...")
    random.seed(2)
    train, test, valid = split_train_test_valid(blocks)

    save_code_blocks(train, OUT_PATH + "/train")
    save_code_blocks(test, OUT_PATH + "/test")
    save_code_blocks(valid, OUT_PATH + "/valid")

    print("Making empty models directory %s..." % MODEL_PATH)
    os.makedirs(MODEL_PATH)


if __name__ == "__main__":
    main()
