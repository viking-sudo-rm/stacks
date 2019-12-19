# Model hyperparameters.
local BATCH_SIZE = 64;
local EMBEDDING_DIM = 8;
local STACK_DIM = 5;
local HIDDEN_DIM = 16;
# local DROPOUT = 0.0;

local NUM_OPS = std.extVar("NUM_OPS");

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "stack" then {
    "type": "stack-encoder",
    "input_dim": EMBEDDING_DIM,
    "stack_dim": STACK_DIM,
    "hidden_dim": HIDDEN_DIM,
  }
  else if ETYPE == "lstm" then {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": HIDDEN_DIM,
    "bidirectional": false,
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "dyck",
  },

  "train_data_path": "/home/willm/data/dyck" + NUM_OPS + "/train.txt",
  "validation_data_path": "/home/willm/data/dyck" + NUM_OPS + "/valid.txt",
  
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

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["source", "num_tokens"]],
      "batch_size": BATCH_SIZE,
  },
  "trainer": {
      "optimizer": {
        "type": "adam",
      },
      "num_epochs": 300,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "-perplexity",
      "num_serialized_models_to_keep": 1,
  }
}