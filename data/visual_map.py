import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
        ax.set_aspect('equal')
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


class ScatterDebugger:
    def __init__(self):
        self.basedir = os.path.dirname(os.path.abspath(__file__))
        tabledir = os.path.join(self.basedir, 'emoji_coordinate2.000.xls')

        self.df = pd.read_excel(tabledir)

        self.text_point = None
        plt.figure()
        self.init_canvas()

    def init_canvas(self):

        plt.xticks(np.arange(-3, 3, 0.1))
        plt.yticks(np.arange(-3, 3, 0.1))

        plt.grid(True, linestyle='-.')

        for index, row in self.df.iterrows():
            picdir = os.path.join(self.basedir, 'emoji/' + str(index) + '.png')
            image_path = get_sample_data(picdir)
            imscatter(row['X'], row['Y'], image_path, zoom=0.1)
            plt.plot(row['X'], row['Y'])

    def show(self):
        plt.show()
        plt.cla()
        self.init_canvas()

    def plot_text(self, x, y):
        plt.scatter(x, y, c='k', label='text')
        self.text_point = (x, y)

    def plot_fused(self, x, y):
        plt.scatter(x, y, c='g', label='fused')
        plt.plot([self.text_point[0], x], [self.text_point[1], y], 'k')

