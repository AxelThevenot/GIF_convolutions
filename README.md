# GIF_convolutions

[GIF CONV2D](GIFS/Input Shape : (3%2C 7%2C 7) - Output Shape : (2%2C 3%2C 3) - K : (3%2C 3) - P : (1%2C 1) - S : (2%2C 2) - D : (2%2C 2) - G : 1.gif)

## Make your own gifs of Conv2d forward pass 

### Download or clone the folder

### Requirements

Run in your bash 
```
pip install -r requirements.txt
```

or 
```
pip install numpy opencv-python scikit-image imageio
```
### Customize your GIF
if you want to customize your GIF :  all the arguments are written at the top of the `main.py` script.
You can change with the values you want :
  - Height of input channels
  - Width of input channels
  - Number of input channels
  - Number of output channels
  - Kernel size
  - Padding 
  - Strides
  - Dilations
  - Groups

If you also want to change the colors. Go to the `pixel.py` script at the top of the `Pixel` class. 
And if you want to change the padding of the image and the margins between Inputs, Kernels and Outputs, you can refer to line 80 of the `conv2d.py` script where the `renderer.get_full_image()` method is called.

## Test the impact of the arguments with Pytorch

You will of course need `Pytorch` for this part. Every script is in the folder `run_test/`. It is highly recommended to have the GPU drivers. If not, the tests will run on the CPU instead of the GPU.
