# Credits to
# https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
# Also to
# CS 168 helper script for Assignment 2
# http://web.stanford.edu/class/cs168/p2.pdf

from colorsys import hls_to_rgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
import numpy as np
from sklearn import manifold, datasets
import torch
from sklearn import manifold, datasets

KNOWN_COLORS = ['blueviolet', 'limegreen', 'darksalmon', 'cadetblue',
'lightgoldenrodyellow', 'chartreuse', 'papayawhip', 'steelblue', 'purple',
'green', 'lawngreen', 'blue', 'darkslategray', 'dodgerblue', 'indigo',
'saddlebrown', 'aquamarine', 'violet', 'lime', 'midnightblue', 'fuchsia',
'snow', 'burlywood', 'mistyrose', 'beige', 'orangered', 'darkviolet',
'mediumorchid', 'antiquewhite', 'lightblue', 'darkgrey', 'lightslategray',
'indianred', 'crimson', 'olive', 'lightsalmon', 'sandybrown', 'chocolate',
'goldenrod', 'white', 'gold', 'tan', 'plum', 'darkgreen', 'darkorange',
'palegoldenrod', 'powderblue', 'greenyellow', 'm', 'mediumvioletred', 'ivory',
'turquoise', 'cornflowerblue', 'aliceblue', 'seagreen']


TSNE_MAX_LENGTH = 5000

def get_distinct_colors(n):
    # Fix seed every time to keep color the same for all drivers
    np.random.seed(10)

    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return np.array(colors)


def get_named_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    select = ['b', 'cyan', 'darkred', 'plum', 'red',
              'green', 'mediumpurple', 'chocolate', 'yellow', 'lightslategrey',
              'lightpink', 'peachpuff', 'brown', 'lime', 'darkviolet']
    result = np.array([colors[b] for b in select])

    return result


#PLOT_COLORS = get_distinct_colors(100)
#PLOT_COLORS = get_named_colors()
PLOT_COLORS = KNOWN_COLORS


def tb_plot_prep():
    fig = Figure(dpi=128)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    return canvas, fig, ax


def tb_plot_wrap_up(canvas, fig, ax):

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height,
                                                                        width,
                                                                        3)
    return image, fig


def tb_plot_decorator(function):
    def wrapper(*args, **kwargs):
        canvas, fig, ax = tb_plot_prep()
        extended_kwargs = kwargs
        extended_kwargs.update({'canvas': canvas, 'ax': ax, 'fig': fig})
        canvas, fig, ax = function(*args, **extended_kwargs)
        return tb_plot_wrap_up(canvas, fig, ax)

    return wrapper


@tb_plot_decorator
def plot_simple(x_values, y_values, legends, **kwargs):

    canvas = kwargs['canvas']
    fig = kwargs['fig']
    ax = kwargs['ax']

    assert len(x_values) == len(y_values)

    for i in range(len(x_values)):
        ax.plot(x_values[i], y_values[i], label=legends[i])

    return canvas, fig, ax


@tb_plot_decorator
def plot_confusion_matrix(cm, color='YlGn', **kwargs):

    canvas = kwargs['canvas']
    fig = kwargs['fig']
    ax = kwargs['ax']

    heatmap = ax.pcolor(cm, cmap=color)
    fig.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(cm.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(cm.shape[1]) + 0.5, minor=False)

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Groud Truth')

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_title('Confusion Matrix')

    return canvas, fig, ax


@tb_plot_decorator
def plot_tsne(embeddings, labels, perplexity=7, n_components=2, **kwargs):

    canvas = kwargs['canvas']
    fig = kwargs['fig']
    ax = kwargs['ax']
    tsne = manifold.TSNE(n_components=n_components, init='pca',
                         random_state=0, perplexity=perplexity)
    
    if len(embeddings) > TSNE_MAX_LENGTH:
        # Logger.get_unique_logger().log(f'Too many embeddings, '
        #                               f'will randomly select {max_length}')

        indices = np.random.randint(0, len(embeddings), TSNE_MAX_LENGTH)
        embeddings = np.array(embeddings)[indices]
        labels = np.array(labels)[indices]

    Y = tsne.fit_transform(embeddings)
    all_colors = [PLOT_COLORS[l] for l in labels]
    ax.scatter(Y[:, 0], Y[:, 1], s=3, c=all_colors)
    plt.tight_layout(pad=7)
    ax.set_title(f't-SNE With Perplexity $={perplexity}$')

    return canvas, fig, ax