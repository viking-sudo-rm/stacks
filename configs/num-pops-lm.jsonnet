# Model hyperparameters.
local EMBEDDING_DIM = 300;
local CHAR_EMBEDDING_DIM = 50;
local INPUT_DIM = EMBEDDING_DIM + CHAR_EMBEDDING_DIM;
local HIDDEN_DIM = 650;
local STACK_DIM = HIDDEN_DIM;
local SUMMMARY_SIZE = 7;
local SUMMARY_DIM = SUMMMARY_SIZE * STACK_DIM;

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local BATCH_SIZE = 16;
local PATIENCE = 10;
local OPTIMIZER = "adam";
local WEIGHT_DECAY = 5e-6;
local CHAR_DROPOUT = 0.5;
local EMBED_DROPOUT = 0.4;
local CONT_DROPOUT = 0.1;

# Parameters about fixing the stack action distribution.
local POPS_WEIGHT = std.extVar("POPS");
local PRIOR_WEIGHT = std.extVar("PRIOR");
# local PRIOR_DISTRIBUTION = [0.72, 0.08, 0.07, 0.03, 0.02];
local PRIOR_DISTRIBUTION = [0.27606, 0.52671, 0.16391, 0.02957, 0.00348, 0.00026, 0.00001];
local REVERSE_TOKENS = true;

# Path to the data on the file system.
local DATA_ROOT = "/net/nfs.corp/allennlp/willm/data";
local DATASET = std.extVar("DATASET");

# Encoder specified by command line arguments.
local ETYPE = std.extVar("ENCODER");
local ENCODER = 
  if ETYPE == "kpop-lstm" then {
    "type": "stack-encoder",
    "stack_dim": STACK_DIM,
    "summary_size": SUMMMARY_SIZE,
    "num_actions": SUMMMARY_SIZE,
    "controller": {
      "type": "suzgun-generic-rnn",
      "rnn_cell_type": "lstm",
      "input_dim": INPUT_DIM,
      "summary_dim": SUMMARY_DIM,
      "hidden_dim": HIDDEN_DIM,
    },
    "dropout": CONT_DROPOUT,
    "project_states": false,
    "store_policies": true,
  }
  else if ETYPE == "kpop-ff" then {
    "type": "stack-encoder",
    "stack_dim": STACK_DIM,
    "summary_size": SUMMMARY_SIZE,
    "num_actions": SUMMMARY_SIZE,
    "controller": {
      "type": "feedforward",
      "input_dim": INPUT_DIM,
      "summary_dim": SUMMARY_DIM,
      "feedforward": {
        "input_dim": INPUT_DIM + SUMMARY_DIM,
        "num_layers": 2,
        "hidden_dims": HIDDEN_DIM,
        "activations": ["relu", "tanh"],
      },
    },
    "dropout": CONT_DROPOUT,
    "project_states": false,
    "store_policies": true,
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "simple_lm",
    "end_token": "<eos>",
    "add_lengths": true,
    "reverse_tokens": REVERSE_TOKENS,
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
    "type": "num-pops-lm",

    "pops_weight": POPS_WEIGHT,
    "prior_weight": PRIOR_WEIGHT,
    "prior_distribution": PRIOR_DISTRIBUTION,

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
      "type": OPTIMIZER,
      "weight_decay": WEIGHT_DECAY,
    },
    "num_epochs": 300,
    "patience": PATIENCE,
    "cuda_device": 0,
    "validation_metric": "-perplexity",
    "num_serialized_models_to_keep": 1,
  }
}
