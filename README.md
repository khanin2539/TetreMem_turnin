# TetreMem_turnin

1. We quantized the model starting from 

Basic Block:
  1. begin convolution layer->batch normalization->relu layer.
  2. perform float functional add at the end of each basic block
 Resnet Block:
  1. Quantize the input layer
  2. continue quantizing from 
    layer 1: convolution layer->batch normalization->relu layer->convolution layer->batch normalization.
    layer 2: convolution layer->batch normalization->relu layer->convolution layer->batch normalization.
ascwawdwd
