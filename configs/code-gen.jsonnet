# Model hyperparameters.
local EMBEDDING_DIM = 64;  # 300;
local INPUT_DIM = EMBEDDING_DIM;
local HIDDEN_DIM = 1100;  # 650;
local STACK_DIM = 128;
local SUMMMARY_SIZE = 3;
local SUMMARY_DIM = SUMMMARY_SIZE * STACK_DIM;
local MAX_DEPTH = 32;

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local BATCH_SIZE = 20;  # 16;
local PATIENCE = 5;
local CHAR_DROPOUT = 0.5;
local EMBED_DROPOUT = 0.5;
# local CONT_DROPOUT = 0.1;
local WEIGHT_DECAY = 1.2e-6;

# TODO: Get a strong baseline LSTM going using https://github.com/salesforce/awd-lstm-lm.

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET = std.extVar("DATASET");

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "lstm" then {
    "type": "lstm",
    "input_size": INPUT_DIM,
    "hidden_size": HIDDEN_DIM,
    "bidirectional": false,
    "num_layers": 1,
  }
  else if ETYPE == "noop" then {
    "type": "stack-encoder",
    "stack_dim": STACK_DIM,
    "summary_size": SUMMMARY_SIZE,
    "stack_type": "noop",
    "num_actions": 3,
    "max_depth": MAX_DEPTH,
    "controller": {
      "type": "suzgun-generic-rnn",
      "rnn_cell_type": "lstm",
      "input_dim": INPUT_DIM,
      "summary_dim": SUMMARY_DIM,
      "hidden_dim": HIDDEN_DIM,
    },
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "python",
    # "strip_names": false,
    # "strip_numbers": false,
    "max_length": 500,
  },

  "train_data_path": DATA_ROOT + "/" + DATASET + "/train",
  "validation_data_path": DATA_ROOT + "/" + DATASET + "/valid",
  
  "model": {
    "type": "language_model",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
      },
    },

    "contextualizer": ENCODER,
    "dropout": EMBED_DROPOUT,

  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["source", "num_tokens"]],
    "batch_size": BATCH_SIZE,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "weight_decay": WEIGHT_DECAY,
    },
    "num_epochs": 300,
    "patience": PATIENCE,
    "cuda_device": 0,
    "validation_metric": "-perplexity",
    "num_serialized_models_to_keep": 1,
  }
}
