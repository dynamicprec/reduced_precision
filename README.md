# A Dynamic Approach to Accelerate Deep Learning Training
## Requirements
* Intel Caffe Framework (PyCaffe compilation) to train ResNet
* Pytorch 1.4 to train the seq2seq model
* PIN Binary Analysis Tool 3.7
* Python 2.7 or Python 3.6 in case of the seq2seq implementation
* An Intel processor with AVX512 support to use the PinTool
* Reduce Imagenet dataset. (200 categories, 256000 training images, 10000 validation images) to train ResNet.

## Run Experiments
To run the experiments is important to have PIN 3.7 installed in order to compile the Pintool. Check the PIN guide to do this. Our pintool is in the pintool folder of this repository. Please after the compilation you will have a library called dynamic.so which you will use to run the DNN models. Each experiment is on his own folder. Additionally a fifo pipe is needed to launch the experiments, this fifo needs to be in the same folder of the .py script:
```
mkfifo fifopipe
```

### Command
To run the program with the binary analysis tool to emulate the BF16
numerical datatype you need to run the following command to use with
pytorch:

```
pin -inline 1 -t ./pintool/dynamic.so -o new_tests.out,fifopipe, -- python ./seq2seq/seq2seq_model.py 10 0.06 ./weight_results.pt log_dyn_attention.out fifopipe 10
```

To run the program with the binary analysis tool with caffe please run
the command below:

```
pin -inline 1 -t ./pintool/dynamic.so -- python ./resnet/resnet_model.py ./resnet/solver_ema_4_1000_batches_msra.prototxt 100 0.04 training1.csv fifopipe
```

