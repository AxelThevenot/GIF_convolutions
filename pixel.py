import numpy as np

class Pixel:
    # size of the represented pixel (in px)
    size = 20
    # set their color and a place to get their padding in the image
    mode = {
             'input':
             {
               'color': np.array([.99, 1, 1]),
               'padding': np.array([0, 0]),
               'is_pad_color': np.array([0.8, 0.6, 1]),
               'highligth_color': np.array([0.2, 0.6, 0.6])
             },
             'output':
             {
               'color': np.array([.9, .9, 1]),
               'padding': np.array([0, 0]),
               'highligth_color': np.array([0.1, 0.6, 0.6])
             },
             'kernel':
             {
               'color': np.array([.99, 1, 1]),
               'padding': np.array([0, 0]),
               'highligth_color': np.array([0.1, 0.6, 0.6])
             }
           }

    # other colors to define
    line_color = np.array([0.4, 0.4, 0.4])
    line_width = 1
    border_color = np.array([0, 0, 0])


    def __init__(self, g, c, h, w, mode, is_pad=False):
        # get the pixel's properties
        self.g = g
        self.c = c
        self.h = h
        self.w = w
        self.mode = mode
        self.is_pad = is_pad

        # stands to set them to be highlighted in the image
        self.highlight = False

        # coordinates (absolute, not in the image)
        self.x = None
        self.y = None

    def set_coord(self, x, y):
        self.x = x
        self.y = y

    def coord(self):
        # get the coordinates in the full inmage
        padx, pady = Pixel.mode[self.mode]['padding']
        return (self.x + padx) * Pixel.size,(self.y + pady) * Pixel.size

    def color(self):
        # return color according to its state/properties
        if self.highlight:
            return Pixel.mode[self.mode]['highligth_color']
        if self.is_pad:
            return Pixel.mode[self.mode]['is_pad_color']
        return Pixel.mode[self.mode]['color']
