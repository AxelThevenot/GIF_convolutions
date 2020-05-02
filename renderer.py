import time
import cv2
import numpy as np

def get_waiting_frame():
    # generate a three dot waiting frame in case of a long waiting time
    waiting_frame = np.ones((256, 512, 3))
    seconds = int(time.time() % 4)
    r = 8
    for second in range(-1, seconds -1):
        dy = r * 5 * second
        cv2.circle(waiting_frame, (256 + dy, 128), r, (.2, .6, .6), -1)
    return waiting_frame


def get_input_img(inputs, input_shape, G):

    # get inputs shape
    C_in, H_in, W_in = input_shape
    n_by_group_in = int(C_in / G)

    # set a padding between the inputs to be visualized
    sx = H_in + 1
    sy = 0

    # compute the image size
    img_w = W_in # size of the input
    img_w += sy * (n_by_group_in - 1)  # number of in -> shifts
    img_h = H_in # size of the output
    img_h += (sx * (n_by_group_in - 1)) # number of in -> shifts
    img_h += 3 * sx * n_by_group_in * (G - 1) // 2 # number of group -> shifts
    bg_shape = (img_h, img_w, 3)

    # initialize the empty image
    img = np.ones(bg_shape)

    # set the pixels a their place regargind the shift and coordinate in input
    for g, group in enumerate(inputs):
        for c, channel in enumerate(group):
            y = sy * c
            x = 3 * g * sx * n_by_group_in // 2  + (sx * c)

            for h, line in enumerate(channel):
                for w, pixel in enumerate(line):
                    img[x+h, y+w] = pixel.color()
                    pixel.set_coord(x+h, y+w)
    return img

def get_output_img(outputs, output_shape, G, shifts=(2, 4)):

    # get outputs shape
    C_out, H_out, W_out = output_shape
    n_by_group_out = int(C_out / G)

    # set a padding between the outputs to be visualized
    sx = H_out + 1
    sy = 0

    # compute the image size
    img_w = W_out # size of the output
    img_w += sy * (n_by_group_out - 1)  # number of out -> shifts
    img_h = H_out # size of the output
    img_h += (sx * (n_by_group_out - 1)) # number of out -> shifts
    img_h += 3 * sx * n_by_group_out * (G - 1) // 2 # number of group -> shifts
    bg_shape = (img_h, img_w, 3)

    # initialize the empty image
    img = np.ones(bg_shape)

    # set the pixels a their place regargind the shift and coordinate in output
    for g, group in enumerate(outputs):
        for c, channel in enumerate(group):
            y = sy * c
            x = 3 * g * sx * n_by_group_out // 2  + (sx * c)

            for h, line in enumerate(channel):
                for w, pixel in enumerate(line):
                    img[x+h, y+w] = pixel.color()
                    pixel.set_coord(x+h, y+w)
    return img


def get_kernel_img(kernels, C_in, C_out, K, G):

    # get kernels shape
    n_by_group_in = int(C_in / G)
    n_by_group_out = int(C_out / G)

    # set a padding between the kernels to be visualized
    sx, sy = K[0] + 1, K[1] + 1

    # compute the image size

    img_w = K[1] # size of the kernel
    img_w += (n_by_group_out - 1) * sy  # number of out -> shifts
    img_h = K[0] # size of the kernel
    img_h += (n_by_group_in  - 1) * sx # number of in -> shifts
    img_h += sx * ((G - 1) * (n_by_group_in + 1)) # number of group -> shifts
    bg_shape = (img_h, img_w, 3)

    # initialize the empty image
    img = np.ones(bg_shape)

    # set the pixels a their place regargind the shift and coordinate in kernels
    for g, group in enumerate(kernels):

        for c_out, channel_out in enumerate(group):
            for c_in, channel in enumerate(channel_out):
                y = sy * c_out
                x = sx * c_in + g * sx * (n_by_group_in + 1)

                for h, line in enumerate(channel):
                    for w, pixel in enumerate(line):

                        pcol = np.array(pixel.color())
                        img[x+h, y+w] = img[x+h, y+w] + pcol
                        pixel.set_coord(x+h, y+w)
    return img

def get_full_img(input_img, kernel_img, output_img, margin=0, pad=(10, 10)):
    pad_h, pad_w = pad
    # get the max height of the images
    H = max(input_img.shape[0], kernel_img.shape[0], output_img.shape[0])
    # compute the total width with margin bewteen images
    W = input_img.shape[1] + kernel_img.shape[1] + output_img.shape[1]
    W = W + 2 * margin
    # initialize the full image
    full_img = np.ones((H + 2 * pad_h, W + 2 * pad_w, 3))

    # place on by one the images by remembering the axis shifts to top left
    padding_memory = []
    w_shift = pad_w
    for img in [input_img, output_img, kernel_img]:
        h, w, _ = img.shape # get the shape
        # center vertically
        top_shift = (H - h) // 2 + pad_h
        # place the image
        full_img[top_shift:top_shift+h,w_shift:w_shift+w] = img

        # actualize
        padding_memory.append([top_shift, w_shift])
        w_shift += w
        w_shift += margin

    return full_img, padding_memory
