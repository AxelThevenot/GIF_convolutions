# GIF_convolutions

Full explanation of the Conv2D :
[Conv2d: Finally Understand What Happens in the Forward Pass](https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148?source=friends_link&sk=ca293c1264492ef1be0cb77d2199b8be)
![GIF CONV2D](GIFS/Input%20Shape%20:%20(3%2C%207%2C%207)%20-%20Output%20Shape%20:%20(2%2C%203%2C%203)%20-%20K%20:%20(3%2C%203)%20-%20P%20:%20(1%2C%201)%20-%20S%20:%20(2%2C%202)%20-%20D%20:%20(2%2C%202)%20-%20G%20:%201.gif)

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
