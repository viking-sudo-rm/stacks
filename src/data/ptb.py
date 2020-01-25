from typing import Dict, Optional, Iterator, Tuple

from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tree import Tree
from overrides import overrides
import os
import pickle

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from src.utils.trees import (left_distances, right_distances, binarize, flatten_tree,
                             get_pos_tags, add_bert_tags, remove_punctuation, add_eos,
                             from_left_distances, from_right_distances)


@DatasetReader.register("ptb-wsj")
class PtbWsjReader(DatasetReader):

    """This class includes many options for label type and input preprocessing.

    The options allow the data to be read in a format like the preprocessing used by Mikolov
    et al. (2011) in their generation of the language modeling corpus.
    """

    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 tokens_field: str = "source",
                 left_tags_field: str = "left_tags",
                 right_tags_field: str = "right_tags",
                 pos_tags_field: str = "pos_tags",
                 left_namespace: str = "labels",
                 right_namespace: str = "labels",
                 pos_namespace: str = "labels",
                 left_tags: bool = True,
                 right_tags: bool = False,
                 pos_tags: bool = False,
                 reverse_sentences: bool = False,
                 min_length: int = 1,
                 bert_tags: bool = False,
                 eos: bool = False,
                 do_lowercase: bool = False,
                 lexicon_path: str = None,
                 strip_numeric: bool = False,
                 remove_punct: bool = False,
                 lazy: bool = False):

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokens_field = tokens_field
        self._min_length = min_length
        self._bert_tags = bert_tags
        self._eos = eos
        self._do_lowercase = do_lowercase
        self._strip_numeric = strip_numeric
        self._remove_punct = remove_punct

        self._left_tags_field = left_tags_field
        self._right_tags_field = right_tags_field
        self._pos_tags_field = pos_tags_field
        self._left_namespace = left_namespace
        self._right_namespace = right_namespace
        self._pos_namespace = pos_namespace
        self._left_tags = left_tags
        self._right_tags = right_tags
        self._pos_tags = pos_tags
        self._reverse_sentences = reverse_sentences
        
        if lexicon_path is not None:
            with open(lexicon_path, "rb") as fh:
                self._lexicon = pickle.load(fh)
                self._lexicon.add("<eos>")
        else:
            self._lexicon = None

    @overrides
    def _read(self, path: str) -> Iterator[Instance]:
        for tree in self._get_trees(path):
            tree, bin_tree = self.get_clean_trees(tree)
            tokens = flatten_tree(bin_tree)
            if len(tokens) < self._min_length:
                # For language modeling, we need more than one word.
                continue

            left_tags = left_distances(bin_tree) if self._left_tags else None
            right_tags = right_distances(bin_tree) if self._right_tags else None
            # FIXME: when should we add bERT tags here?
            pos_tags = get_pos_tags(tree, self._bert_tags, self._eos) if self._pos_tags else None
            yield self.text_to_instance(tokens, left_tags, right_tags, pos_tags)

    @overrides
    def text_to_instance(self, tokens, left_tags=None, right_tags=None, pos_tags=None):

        if self._reverse_sentences:
            # Option to reverse tokens and all co-indexed label sequences.
            tokens = list(reversed(tokens))
            left_tags = left_tags and list(reversed(left_tags))
            right_tags = right_tags and list(reversed(right_tags))
            pos_tags = pos_tags and list(reversed(pos_tags))

        tokens = [self._cased_token(token) for token in tokens]
        tokens_field = TextField(tokens, self._token_indexers)
        instance = {self._tokens_field: tokens_field}

        if left_tags is not None:
            left_tags = [str(tag) for tag in left_tags]
            left_tags_field = SequenceLabelField(left_tags,
                                                 sequence_field=tokens_field,
                                                 label_namespace=self._left_namespace)
            instance[self._left_tags_field] = left_tags_field

        if right_tags is not None:
            right_tags = [str(tag) for tag in right_tags]
            right_tags_field = SequenceLabelField(right_tags,
                                                  sequence_field=tokens_field,
                                                  label_namespace=self._right_namespace)
            instance[self._right_tags_field] = right_tags_field

        if pos_tags is not None:
            pos_tags = [str(tag) for tag in pos_tags]
            pos_tags_field = SequenceLabelField(pos_tags,
                                                sequence_field=tokens_field,
                                                label_namespace=self._pos_namespace)
            instance[self._pos_tags_field] = pos_tags_field

        return Instance(instance)

    def get_clean_trees(self, tree: Tree) -> Tuple[Tree, list]:
        """Make modifications to trees; add/remove tokens."""
        if self._remove_punct:
            tree = remove_punctuation(tree)

        bin_tree = binarize(tree)
        if self._bert_tags:
            # Add [CLS] and [SEP] tokens to the tree.
            bin_tree = add_bert_tags(bin_tree)
        elif self._eos:
            # Add end of line constituent to the tree.
            bin_tree = add_eos(bin_tree)

        return tree, bin_tree

    def _cased_token(self, token: str) -> Token:
        """Change the representation of tokens."""
        token = token.lower() if self._do_lowercase else token
        if self._strip_numeric and token.isnumeric():
            return Token("N")
        elif self._lexicon is not None and token not in self._lexicon:
            return Token("<unk>")
        else:
            return Token(token)

    @staticmethod
    def _get_trees(path: str) -> Iterator[list]:
        """Takes either a directory of .mrg files or a single .txt file."""
        if os.path.isdir(path):
            fileids = r"wsj_.*\.mrg"
            reader = BracketParseCorpusReader(path, fileids)
            yield from reader.parsed_sents()
        else:
            with open(path) as fh:
                for line in fh.read().split("\n\n"):
                    yield Tree.fromstring(line)

    def get_binary_trees(self, path: str) -> Iterator[list]:
        """Get gold-standard binary trees for a dataset.

        Importantly, this method should guarantee that the order of sentences is the same as what
        the model sees at prediction time.
        """
        for instance in self.read(path):
            tokens = [tok.text for tok in instance[self._tokens_field]]
            if self._left_tags:
                tags = [int(d) for d in instance[self._left_tags_field]]
                yield from_left_distances(tokens, tags)
            elif self._right_tags:
                tags = [int(d) for d in instance[self._right_tags_field]]
                yield from_right_distances(tokens, tags)
            else:
                raise ValueError("Can't get binary trees for a reader without left/right tags.")


def write_corpus_file(dirname):
    """Takes all .mrg PTB files in a directory and puts them in a single file.
    This allows for faster retrieval from disk."""
    fileids = r"wsj_.*\.mrg"
    reader = BracketParseCorpusReader(dirname, fileids)
    text = "\n\n".join(str(tree) for tree in reader.parsed_sents())
    
    filename = dirname + ".txt"
    with open(filename, "w") as fh:
        fh.write(text)


def build_corpus_dir_from_secs(dirname, sec_iter):
    """Partition according to Mikolov et al., 2011.
    Example args:
      - "train", range(0, 21)
      - "dev", range(21, 23)
      - "test", range(23, 25)
    """
    for sec in sec_iter:
        os.system("cp %.2d/*.mrg %s/" % (sec, dirname))
