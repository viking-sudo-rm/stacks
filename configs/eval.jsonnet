# Model hyperparameters.
local EMBEDDING_DIM = 16;
local INPUT_DIM = EMBEDDING_DIM;
local HIDDEN_DIM = 32;
local STACK_DIM = HIDDEN_DIM;
local SUMMMARY_SIZE = 5;
local SUMMARY_DIM = SUMMMARY_SIZE * STACK_DIM;

# Optimization hyperparameters.
# Refer to https://github.com/viking-sudo-rm/bert-parsing/blob/master/configs/language-modeling/ptb.jsonnet
local BATCH_SIZE = 16;
local PATIENCE = 5;
local OPTIMIZER = "adam";
# local WEIGHT_DECAY = 5e-6;
# local CHAR_DROPOUT = 0.5;
# local EMBED_DROPOUT = 0.4;
# local CONT_DROPOUT = 0.1;

# Hyperparameters for setting a prior on the stack action distribution, computed from corpus.
local POPS_WEIGHT = 0.0;
local PRIOR_WEIGHT = std.extVar("PRIOR");
local REVERSE_TOKENS = false;
local PRIOR_DISTRIBUTION =
  if REVERSE_TOKENS then [
    # We use right distances if the sequence is reversed.
    0.40770, 0.19960, 0.39270
  ]
  else [
    # We use left distances otherwise.
    0.59617, 0.10801, 0.15048, 0.07347, 0.03679
  ];

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
  else if ETYPE == "kpop-lstm" then {
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
    "project_states": false,
    "store_policies": true,
  }
  else if ETYPE == "kpush-lstm" then {
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
    "project_states": false,
    "multipush": true,
  }
  else if ETYPE == "kpush-ff" then {
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
    "project_states": false,
    "multipush": true,
  }
  else error "Invalid encoder: " + std.manifestJson(ETYPE);


{
  "dataset_reader": {
    "type": "eval",
    "add_lengths": true,
  },

  # Sample independently from the same distribution. Testing generalization is a different matter.
  "train_data_path": "50000:100:2",
  "validation_data_path": "5000:100:3",  # This was 200 before.
  
  "model": {
    "type": "policy-lm",

    "pops_weight": POPS_WEIGHT,
    "prior_weight": PRIOR_WEIGHT,
    "prior_distribution": PRIOR_DISTRIBUTION,

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
      "num_epochs": 500,
      "patience": PATIENCE,
      "cuda_device": 0,
      "validation_metric": "-perplexity",
      "num_serialized_models_to_keep": 1,
  }
}