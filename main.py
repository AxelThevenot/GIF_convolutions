import cv2
import numpy as np


from pixel import Pixel
from conv2d import Conv2DVisualizer
import renderer


C_in = 2         # number of input channels
C_out = 4           # number of output channels
H_in = 7            # weight of the inputs
W_in = 7           # width of the inputs
K = 3              # kernel size (can be an integer of a two-value-integer tuple)
P = 2               # padding  (can be an integer of a two-value-integer tuple)
S = 2              # stride   (can be an integer of a two-value-integer tuple)
D = 1              # dilation (can be an integer of a two-value-integer tuple)
G = 2               # number of group

time_sleep = 0.3    # time in second between two images


if __name__ == '__main__':

    # if not specify, understand by default the same value on each axis
    if isinstance(K, int):
        K = (K, K)
    if isinstance(P, int):
        P = (P, P)
    if isinstance(S, int):
        S = (S, S)
    if isinstance(D, int):
        D = (D, D)

    # the kernel must have a positive size
    assert K[0] > 0 and K[1] > 0 and K[0] <= H_in + P[0] and K[1] <= W_in + P[1]
    # paddings must be posititve
    assert P[0] >= 0 and P[1] >= 0
    # strides must have a non-zero positive value
    assert S[0] > 0 and S[1] > 0
    # dilations must have a non-zero positive value
    assert D[0] > 0 and D[1] > 0
    # the number of channel in and out must be divisible by the number of groups
    assert G > 0 and C_in % G == 0 and C_out % G == 0



    # the resulting output must have a non-zero positive size
    assert H_out > 0 and W_out > 0

    # create inputs pixels checking if they are padding pixels
    is_pad = lambda h, w: h<P[0] or w<P[1] or h>=H_in+P[0] or w>=W_in+P[1]
    inputs = [[[[Pixel(g, c, h, w, 'input', is_pad=is_pad(h, w))
                            for w in range(W_in + 2 * P[1])]
                            for h in range(H_in + 2 * P[0])]
                            for c in range(C_in // G)]
                            for g in range(G)]

    # create outputs and kernels the same way
    outputs = [[[[Pixel(g, c, h, w, 'output')
                            for w in range(W_out)]
                            for h in range(H_out)]
                            for c in range(C_out // G)]
                            for g in range(G)]

    kernels = [[[[[Pixel(g, c_in, h, w, 'kernel')
                            for w in range(K[1])]
                            for h in range(K[0])]
                            for c_in in range(C_in // G)]
                            for c_out in range(C_out // G)]
                            for g in range(G)]

    # convert them to np array to be easily manipulate
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    kernels = np.array(kernels)

    # initialize an OpenCV's normal window
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # initialize the Conv2D visualizer with our parameters
    conv2d = Conv2DVisualizer(inputs, kernels, outputs, C_in, C_out, H_in, W_in, H_out, W_out, K, P, S, D, G, time_sleep)
    conv2d.start()  # start the thread

    while True:

        # get the image of the Conv2D forward pass state
        img = conv2d.img

        # the first image can take a while to be created
        if img is None:
            img = renderer.get_waiting_frame()

        # convert to bytes and BGR to be display by OpenCV
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Frame', img)
        # quit by pressing the 'q' key
        if cv2.waitKey(1) & 0xff == ord('q'):
            conv2d.stop() # stop the thread
            break
