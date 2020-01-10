# Model hyperparameters.
local EMBEDDING_DIM = 100;
local HIDDEN_DIM = 100;
local STACK_DIM = 64;
local SUMMMARY_SIZE = 5;
local SUMMARY_DIM = SUMMMARY_SIZE * STACK_DIM;

# Optimization hyperparameters.
local BATCH_SIZE = 16;
local PATIENCE = 10;
local DROPOUT = 0.5;
local EMBED_DROPOUT = 0.5;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "lstm" then {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": HIDDEN_DIM,
    "bidirectional": false,
    "dropout": DROPOUT,
  }
  else if ETYPE == "dmg" then {
    "type": "minimalist-grammar",
    "stack_dim": STACK_DIM,
    "summary_size": SUMMMARY_SIZE,
    "controller": {
      "type": "suzgun-rnn",
      "input_dim": EMBEDDING_DIM,
      "summary_dim": SUMMARY_DIM,
      "hidden_dim": HIDDEN_DIM,
      "dropout": DROPOUT,
    },
  }
  else if ETYPE == "dmg-ff" then {
    "type": "minimalist-grammar",
    "stack_dim": STACK_DIM,
    "summary_size": SUMMMARY_SIZE,
    "controller": {
      "type": "feedforward",
      "input_dim": EMBEDDING_DIM,
      "summary_dim": SUMMARY_DIM,
      "feedforward": {
        "input_dim": EMBEDDING_DIM + SUMMARY_DIM,
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
    "summary_size": SUMMMARY_SIZE,
    "controller": {
      "type": "suzgun-generic-rnn",
      "rnn_cell_type": "lstm",
      "input_dim": EMBEDDING_DIM,
      "summary_dim": SUMMARY_DIM,
      "hidden_dim": HIDDEN_DIM,
      "dropout": DROPOUT,
    },
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "agreement",
  },

  "train_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.train",
  "validation_data_path": DATA_ROOT + "/rnn_agr_simple/numpred.val",
  
  "model": {
    "type": "basic_classifier",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
      },
    },

    "seq2seq_encoder": ENCODER,

    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": HIDDEN_DIM,
    },

    "dropout": EMBED_DROPOUT,

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size": BATCH_SIZE,
  },
  "trainer": {
      "optimizer": {
        "type": "adam",
      },
      "num_epochs": 300,
      "patience": PATIENCE,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 1,
  }
}
