import time
from threading import Thread

import cv2
import imageio
import numpy as np
from skimage import transform

import renderer
from pixel import Pixel


class Conv2DVisualizer(Thread):

    def __init__(self, inputs, kernels, outputs, C_in, C_out, H_in, W_in, H_out, W_out, K, P, S, D, G, time_sleep):
        Thread.__init__(self)
        self.is_running = False
        # set arguments
        self.inputs = inputs
        self.kernels = kernels
        self.outputs = outputs
        self.C_in = C_in
        self.C_out = C_out
        self.H_in = H_in
        self.W_in = W_in
        self.H_out = H_out
        self.W_out = W_out
        self.K = K
        self.P = P
        self.S = S
        self.D = D
        self.G = G

        # fps
        self.time_sleep = time_sleep

        # set to highlight state when convolved
        self.highlighted_in_out = []
        self.highlighted_kernel = []

        # set the corners to draw lines next
        self.inputs_corners = []
        self.kernel_corners = []
        self.outputs_corners = []

        # will be the grid to delimit the pixels
        self.grid = None
        # will be the animation
        self.img = None

        # to save GIFs
        self.gif_saved = False
        self.images = []



    def corners(self):
        return self.inputs_corners , self.kernel_corners, self.outputs_corners

    def save_gif(self):
        # to save only once
        if not self.gif_saved:
            filename = 'GIFS/'
            filename += f'Input Shape : ({self.C_in}, {self.H_in}, {self.W_in})'
            filename += f' - Output Shape : ({self.C_out}, {self.H_out}, {self.W_out})'
            filename += f' - K : {self.K} - P : {self.P} - S : {self.S}'
            filename += f' - D : {self.D} - G : {self.G}.gif'
            self.images = [(x * 255).astype(np.uint8) for x in self.images]
            imageio.mimsave(filename, self.images, duration=self.time_sleep)
            self.images = []
            self.gif_saved = True

    def set_image(self):
        # # the current state of the conv2D
        input_img = renderer.get_input_img(self.inputs, (self.C_in, self.H_in + 2 * self.P[0], self.W_in + 2 * self.P[1]), self.G)
        output_img = renderer.get_output_img(self.outputs, (self.C_out, self.H_out, self.W_out), self.G)
        kernel_img = renderer.get_kernel_img(self.kernels, self.C_in, self.C_out, self.K, self.G)

        # concatenate them
        full_img, padding_memory = renderer.get_full_img(input_img, output_img, kernel_img, 5, (2, 10))

        # set the shift of the pixels in the image
        Pixel.mode['input']['padding'] = padding_memory[0]
        Pixel.mode['kernel']['padding'] = padding_memory[1]
        Pixel.mode['output']['padding'] = padding_memory[2]

        img = full_img.copy()

        h, w, _ = img.shape

        # resize without interpolation
        img = transform.resize(img,
                               (h * Pixel.size, w * Pixel.size),
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               order=0)

        # set the grid to delimit the pixel or create it if not exists
        if self.grid is None:
            self.grid = np.zeros(img.shape)
            mid = Pixel.size // 2
            for i in range(mid, (h - 1) * Pixel.size + mid, Pixel.size):
                for j in range(mid, (w - 1) * Pixel.size + mid, Pixel.size):
                    if np.sum(img[i, j]) != 3:
                        cv2.rectangle(self.grid, (j - mid, i - mid), (j + mid, i + mid), (1, 1, 1), 2)
        img[np.where(self.grid[...,0])] = Pixel.border_color

        # draw lines
        shift_corners = [[0, 0], [0, Pixel.size], [Pixel.size, 0], [Pixel.size, Pixel.size]]
        shifts = [[0, 0], [0, Pixel.size], [Pixel.size, 0], [Pixel.size, Pixel.size]]
        in_corn, k_corn, out_corn = self.corners()
        for i in range(4):
            sx, sy = shifts[i]
            for j in range(len(k_corn[i])):
                inx, iny = in_corn[i][j].coord()
                kx, ky = k_corn[i][j].coord()
                outx, outy = out_corn[j // (self.C_in // self.G)].coord()

                cv2.line(img, (iny + sy, inx + sx), (ky   + sy, kx   + sx), tuple(Pixel.line_color), Pixel.line_width)
                cv2.line(img, (ky  + sy, kx  + sx), (outy + sy, outx + sx), tuple(Pixel.line_color), Pixel.line_width)

        self.img = img
        if not self.gif_saved:
            self.images.append(self.img)





    def run(self):
        # start the thread
        self.is_running = True
        while self.is_running:
            # for each output channel, process the convolution
            for c_out in range(self.C_out//self.G):

                # highlight the cureent kernels
                for pixel in self.highlighted_kernel:
                    pixel.highlight = False
                self.highlighted_kernel = []

                # affect the pixel implicated in the current conv product
                # and get the corners coordinates to draw lines
                top_left = []
                top_right = []
                bottom_left = []
                bottom_right = []
                for group in self.kernels:
                    kernels_in = group[c_out]
                    for kernel in kernels_in:
                        for i, line in enumerate(kernel):
                            for j, pixel in enumerate(line):
                                self.highlighted_kernel.append(pixel)
                                pixel.highlight = True
                                if i == 0 and j == 0:
                                    top_left.append(pixel)
                                if i == 0 and j == self.K[1] -1:
                                    top_right.append(pixel)
                                if i == self.K[0] -1 and j == 0:
                                    bottom_left.append(pixel)
                                if i == self.K[0] -1 and j == self.K[1] -1:
                                    bottom_right.append(pixel)
                self.kernel_corners = [top_left, top_right, bottom_left, bottom_right]

                # same for the current output pixels
                for out_h, h in enumerate(range(0, 1 + self.H_in  + 2 * self.P[0] - self.K[0] - (self.D[0] - 1) * (self.K[0] - 1), self.S[0])):
                    for out_w, w in enumerate(range(0, 1 + self.W_in  + 2 * self.P[1] - self.K[1] - (self.D[1] - 1) * (self.K[1] - 1), self.S[1])):

                        for pixel in self.highlighted_in_out:
                            pixel.highlight = False
                        self.highlighted_in_out = []


                        top_left = []
                        top_right = []
                        bottom_left = []
                        bottom_right = []
                        for i, dh in enumerate(range(0, self.K[0] + (self.D[0] - 1) * (self.K[0] - 1), self.D[0])):
                            for j, dw in enumerate(range(0, self.K[1] + (self.D[1] - 1) * (self.K[1] - 1), self.D[1])):

                                for group in self.inputs:
                                    for channel in group:
                                        pixel = channel[h + dh, w + dw]
                                        self.highlighted_in_out.append(pixel)
                                        pixel.highlight = True
                                        if i == 0 and j == 0:
                                            top_left.append(pixel)
                                        if i == 0 and j == self.K[1] -1:
                                            top_right.append(pixel)
                                        if i == self.K[0] -1 and j == 0:
                                            bottom_left.append(pixel)
                                        if i == self.K[0] -1 and j == self.K[1] -1:
                                            bottom_right.append(pixel)
                        self.inputs_corners = [top_left, top_right, bottom_left, bottom_right]

                        outputs_corners = []
                        for group in self.outputs:
                            pixel = group[c_out, out_h, out_w]
                            self.highlighted_in_out.append(pixel)
                            pixel.highlight = True
                            outputs_corners.append(pixel)

                        self.outputs_corners = outputs_corners
                        self.set_image()
                        time.sleep(self.time_sleep)
            self.save_gif()

    def stop(self):
        # stop the thread
        self.is_running = False
