

## model hyper parameters

# input data type: yeast or human
data_type = 'yeast'

# the input feature dimension yeast:6400 human:18362
# input_dim = 18362
input_dim = 6400

# the hidden dimension for view-specific channel
hidden_dim = 512

# the hidden dimension for consensus channel
common_dim = 32

# number of iterations
num_epoch = 1200

# layer aggregation function: max mean concat none
layer_agg = 'mean'

# number of layers
num_layer = 3

# learning rate
learning_rate = 0.005

# dropout rate
dropout_rate = 0.1

# balance factor
beta = 1.0


# train patience for early stop
patience = 35
