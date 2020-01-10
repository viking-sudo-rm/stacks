# Model hyperparameters.
local EMBEDDING_DIM = 300;
local CHAR_EMBEDDING_DIM = 50;
local SUMMARY_SIZE = 2;
local HIDDEN_DIM = 650;
local FEATURE_DIM = EMBEDDING_DIM + CHAR_EMBEDDING_DIM;
local STACK_DIM = FEATURE_DIM;
local SUMMARY_DIM = STACK_DIM * SUMMARY_SIZE;
local HARD = std.extVar("HARD");

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local BATCH_SIZE = 16;
local PATIENCE = 10;
local CHAR_DROPOUT = 0.5;
local DROPOUT = 0.5;
local WEIGHT_DECAY = 1.2e-6;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";

# Encoder specified by command line arguments.
local CTYPE = std.extVar("CTYPE");
local CONTROLLER = 
  if CTYPE == "dmg" then {
    "type": "suzgun-rnn",
    "input_dim": FEATURE_DIM,
    "summary_dim": SUMMARY_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
  }
  else if CTYPE == "dmg-ff" then {
    "type": "feedforward",
    "input_dim": FEATURE_DIM,
    "summary_dim": SUMMARY_DIM,
    "feedforward": {
      "input_dim": FEATURE_DIM + SUMMARY_DIM,
      "num_layers": 2,
      "hidden_dims": HIDDEN_DIM,
      "activations": ["relu", "tanh"],
      "dropout": DROPOUT,
    }
  }
  else if CTYPE == "dmg-lstm" then {
    "type": "suzgun-generic-rnn",
    "rnn_cell_type": "lstm",
    "input_dim": FEATURE_DIM,
    "summary_dim": SUMMARY_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
  }
  else error "Invalid encoder: " + std.manifestJson(CTYPE);


{
  "dataset_reader": {
    "type": "ud-arc-standard",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      },
      "characters": {
        "type": "characters",
      },
    },
  },

  "train_data_path": DATA_ROOT + "/UD_English-EWT/en_ewt-ud-train.conllu",
  "validation_data_path": DATA_ROOT + "/UD_English-EWT/en_ewt-ud-dev.conllu",
  
  "model": {
    "type": "transition-parser",
    "summary_size": SUMMARY_SIZE,
    "hard": HARD,

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
        "characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 8,
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 8,
            "num_filters": CHAR_EMBEDDING_DIM,
            "ngram_filter_sizes": [5],
          },
          "dropout": CHAR_DROPOUT,
        },
      },
    },

    "controller": CONTROLLER,

  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
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
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 1,
  }
}