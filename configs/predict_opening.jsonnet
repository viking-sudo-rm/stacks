# Task settings.
local TASK = "predict_opening";
local OP_RANGE = 5;
local CENTER_EMBED_PROB = 0.8;

# Model hyperparameters.
local BATCH_SIZE = 64;
local EMBEDDING_DIM = 8;
local STACK_DIM = 3 * OP_RANGE;  # Scale the size of the stack with how many parentheses we have.
local HIDDEN_DIM = 16;
# local DROPOUT = 0.0;

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "stack" then {
    "type": "stack-encoder",
    "stack_dim": STACK_DIM,
    "controller": {
      "input_dim": EMBEDDING_DIM + STACK_DIM,
      "num_layers": 2,
      "hidden_dims": HIDDEN_DIM,
      "activations": "relu",
    }
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
    "type": TASK,
    "op_range": OP_RANGE,
    "center_embed_prob": CENTER_EMBED_PROB,
  },

  "train_data_path": "10000:6",
  "validation_data_path": "1000:10",
  
  "model": {
    "type": "simple_tagger",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM,
        },
      },
    },

    "encoder": ENCODER,

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
      "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 2
      },
      "num_epochs": 300,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 1,
  }
}