# Model hyperparameters.
local EMBEDDING_DIM = 300;
local CHAR_EMBEDDING_DIM = 50;
local STACK_DIM = 128;
local HIDDEN_DIM = 650;
local NUM_LAYERS = 1;

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local BATCH_SIZE = 16;
local PATIENCE = 10;
local CHAR_DROPOUT = 0.5;
local EMBED_DROPOUT = 0.5;
local DROPOUT = 0.5;
local WEIGHT_DECAY = 1.2e-6;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET = std.extVar("DATASET");

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "basic" then {
    "type": "stack-encoder",
    "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
    "stack_dim": STACK_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "lstm_controller": false,
  }
  else if ETYPE == "multipop" then {
    "type": "stack-encoder",
    "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
    "stack_dim": STACK_DIM,
    "hidden_dim": HIDDEN_DIM,
    "stack_type": "multipop",
    "summary_size": 6,
    "dropout": DROPOUT,
    "lstm_controller": false,
  }
  else if ETYPE == "lstm" then {
    "type": "lstm",
    "input_size": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
    "hidden_size": HIDDEN_DIM,
    "bidirectional": false,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
  }
  else if ETYPE == "dmg" then {
    "type": "minimalist-grammar",
    "stack_dim": STACK_DIM,
    "controller": {
      "type": "suzgun-rnn",
      "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
      "summary_dim": 2 * STACK_DIM,
      "hidden_dim": HIDDEN_DIM,
      "dropout": DROPOUT,
    },
  }
  else if ETYPE == "dmg-ff" then {
    "type": "minimalist-grammar",
    "stack_dim": STACK_DIM,
    "controller": {
      "type": "feedforward",
      "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
      "summary_dim": 2 * STACK_DIM,
      "feedforward": {
        "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM + 2 * STACK_DIM,
        "num_layers": 2,
        "hidden_dims": HIDDEN_DIM,
        "activations": ["relu", "tanh"],
        "dropout": DROPOUT,
      }
    },
  }
  else if ETYPE == "dmg-lstm" then {
    "type": "minimalist-grammar",
    "stack_dim": STACK_DIM,
    "controller": {
      "type": "suzgun-generic-rnn",
      "rnn_cell_type": "lstm",
      "input_dim": EMBEDDING_DIM + CHAR_EMBEDDING_DIM,
      "summary_dim": 2 * STACK_DIM,
      "hidden_dim": HIDDEN_DIM,
      "dropout": DROPOUT,
    },
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "simple_lm",
    "end_token": "<eos>",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
      },
      "characters": {
        "type": "characters",
      },
    },
  },

  "train_data_path": DATA_ROOT + "/" + DATASET + "/train.txt",
  "validation_data_path": DATA_ROOT + "/" + DATASET + "/valid.txt",
  
  "model": {
    "type": "language_model",

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