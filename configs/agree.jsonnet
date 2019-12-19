# Model hyperparameters.
local EMBEDDING_DIM = 8;
local STACK_DIM = 5;
local HIDDEN_DIM = 16;

# Optimization hyperparameters.
local BATCH_SIZE = 64;
local PATIENCE = 10;
local DROPOUT = 0.5;
local EMBED_DROPOUT = 0.5;

# Encoder specified by command line arguments.
local LSTM_CONTROLLER = std.extVar("LSTM");
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "basic" then {
    "type": "stack-encoder",
    "input_dim": EMBEDDING_DIM,
    "stack_dim": STACK_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "lstm_controller": LSTM_CONTROLLER,
  }
  else if ETYPE == "multipop" then {
    "type": "stack-encoder",
    "input_dim": EMBEDDING_DIM,
    "stack_dim": STACK_DIM,
    "hidden_dim": HIDDEN_DIM,
    "stack_type": "multipop",
    "summary_size": 6,
    "dropout": DROPOUT,
    "lstm_controller": LSTM_CONTROLLER,
  }
  else if ETYPE == "lstm" then {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": HIDDEN_DIM,
    "bidirectional": false,
    "dropout": DROPOUT,
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "agreement",
  },

  "train_data_path": "/home/willm/data/rnn_agr_simple/numpred.train",
  "validation_data_path": "/home/willm/data/rnn_agr_simple/numpred.val",
  
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