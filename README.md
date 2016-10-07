# Adaptive Softmax
The adaptive-softmax project implements an efficient softmax approximation for graphical processing units (GPU).
This method is useful for training language models with large vocabularies.

## Dependencies
This project depends on the following packages:
- [cutorch](https://github.com/torch/cutorch)
- [cunn](https://github.com/torch/cunn)
- [cudnn](https://github.com/soumith/cudnn.torch)
- [torch-tds](https://github.com/torch/tds)
- [torchnet](https://github.com/torchnet/torchnet)
- [torch-rnnlib](https://github.com/facebookresearch/torch-rnnlib)
- [penlight](https://github.com/stevedonovan/Penlight)

## Usage

In order to train a recurrent neural network language model, run

```
> ./cuth train_big_lstm.lua -nhid 512 -isz 512 -dropout 0.25 -batchsize 128 -usecudnn -clip 0.1 -data DATA_DIR
```

where `DATA_DIR` is a directory containing three text files, `train.txt`, `valid.txt` and `test.txt`.

## Contributing

See the CONTRIBUTING file for how to help out.

## License

adaptive-softmax is BSD-licensed. We also provide an additional patent grant.
