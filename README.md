# Adaptive Softmax
The adaptive-softmax project is a Torch implementation of the efficient softmax
approximation for graphical processing units (GPU), described in the paper
"Efficient softmax approximation for GPUs" (http://arxiv.org/abs/1609.04309).

This method is useful for training language models with large vocabularies.
We provide a script to train large recurrent neural network language models,
in order to reproduce the results of the paper.

## Dependencies
This project depends on the following packages:
- [cutorch](https://github.com/torch/cutorch)
- [cunn](https://github.com/torch/cunn)
- [cudnn](https://github.com/soumith/cudnn.torch)
- [torch-tds](https://github.com/torch/tds)
- [torchnet](https://github.com/torchnet/torchnet)
- [torch-rnnlib](https://github.com/facebookresearch/torch-rnnlib)
- [penlight](https://github.com/stevedonovan/Penlight)

## Examples
In order to train a recurrent neural network language model with default
parameters, run

```
th train_big_lstm.lua -data DATA_DIR
```

where `DATA_DIR` is a directory containing three text files, `train.txt`,
`valid.txt` and `test.txt`.

### Penn TreeBank

In order to train a language model on PTB, run the command

```
th train_big_lstm.lua -data PATH/TO/PTB -nhid 512 -isz 512 -dropout 0.5 -usecudnn -cutoff 2000
```

### Text8

In order to train a language model on text8, run the command

```
th train_big_lstm.lua -data PATH/TO/TEXT8 -nhid 512 -isz 512 -dropout 0.25 -batchsize 128 -usecudnn -cutoff 2000,10000
```

### Billion word benchmark

In order to train a language model on the billion word benchmark,
run the command

```
th train_big_lstm.lua -data PATH/TO/BILLION/WORD -nhid 2048 -isz 256 -dropout 0.01 -batchsize 128 -testbatchsize 128 -threshold 2 -usecudnn -cutoff 4000,40000,200000
```

## Usage

We now briefly discuss how to use the adaptive softmax in your own projects.
We provide a Torch layer called `nn.AdaptiveSoftMax` and a corresponding
criterion, called `nn.AdaptiveLoss`, which must be used when training with
the adaptive softmax. The vocabulary must be sorted by decreasing frequency,
so that frequent words correspond to small indices.

The constructor of the `nn.AdaptiveSoftMax` layer takes two arguments:
`hidden_size`, which is the size of the input of the adaptive softmax
and `cutoff`, which is a table indicating the limits of the different clusters.
The constructor of the `nn.AdaptiveLoss` criterion takes as only argument the
`cutoff` table.

```lua
local nword       = 44372
local hidden_size = 256
local cutoff      = { 2000, 10000, nword }

local decoder   = nn.AdaptiveSoftMax( hidden_size, cutoff )
local criterion = nn.AdaptiveLoss( cutoff )
```

In the previous example, we created an adaptive softmax with three clusters.
The first cluster contains the words from 1 to 2000, the second cluster
contains the words from 2001 to 10,000 and finally, the last cluster contains
the word from 10,001 to `nword`.

The `forward` method of the adaptive softmax takes a 2D tensor as input, and
output a table of 2D tensors of scores for each cluster (one tensor per
cluster). In order to be efficient, the `nn.AdaptiveSoftMax` does not compute
the scores for all the word of the vocabulary for all the examples.It is thus
necessary to call the method `setTarget` of the `AdaptiveSoftMax` layer before
each forward pass:

```lua
decoder:setTarget( target )
```

where target is a 1D tensor. This ensure that the adaptive softmax will compute
the scores for the corresponding targets. It is also possible to call the method
`getLogProb`, which computes the log probabilities for all the words of the
vocabulary, given a 2D tensor of hidden vectors.

## Contributing

See the CONTRIBUTING file for how to help out.

## License

adaptive-softmax is BSD-licensed. We also provide an additional patent grant.
