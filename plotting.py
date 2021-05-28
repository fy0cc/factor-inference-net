import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from scipy.stats import pearsonr
import torch


def scatter_image(x, y, **kwargs):
    """
    make a scatter plot and return the figure object
    :param x:  1-d tensor
    :param y:  1-d tensor
    :return:  image in tensor format
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    if "s" in kwargs:
        ax.scatter(x, y, s=kwargs["s"])
    else:
        ax.scatter(x, y)
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])
    if "xlim" in kwargs:
        ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(*kwargs["ylim"])
    if "legend" in kwargs:
        plt.legend(kwargs["legend"])
    # compute pearson correlation and show as legend
    # pr = pearsonr(x, y)
    # plt.legend(["Pearson R:{}".format(pr[0])])
    ax.set_aspect("equal")
    # buf = io.BytesIO()
    if "output" in kwargs:
        plt.savefig(kwargs["output"], format="png")
        plt.close(fig)
        return None
    # plt.close()
    return fig

