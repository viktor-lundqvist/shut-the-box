import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Value_Iteration_Prioritized_Sweeping as vips


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=8, **kw)
            texts.append(text)

    return texts

def rr_graph(pi_names, scoring, title):
    scores, heads = vips.round_robin(pi_names, scoring)
    scores = np.array(scores) * 100
    heads = np.array(heads) * 100
    
    
    fig, ax = plt.subplots()
    
    im, cbar = heatmap(heads, pi_names, pi_names, ax=ax,
                       cmap="binary", cbarlabel="Vinstprocent (%)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig("data.png", dpi=1200)
    plt.show()

"""
pi_names = ["optimal_s", "optimal_d", "digital_g100", "sum_g100", "human", "anti_human", "random_0", "jackpot"]
comps = vips.full_comparison(pi_names)
comps = np.array(comps) * 100


fig, ax = plt.subplots()

im, cbar = heatmap(comps, pi_names, pi_names, ax=ax,
                   cmap="binary", cbarlabel="Likhet (%)")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

plt.title("Likhetsprocent för policyer")
fig.tight_layout()
plt.savefig("data.png", dpi=1200)
plt.show()
"""

pi_names = ["optimal_s", "sum_g100"]
pi_terms = [vips.pickle_load(f"./Terminals/{name}.pickle") for name in pi_names]

n_terminals = 46

y0 = [0] * 46
y1 = [0] * 46

for pos, p in pi_terms[0].items():
    y0[vips.pos_tile_sum(pos)] += p

for pos, p in pi_terms[1].items():
    y1[vips.pos_tile_sum(pos)] += p          

x = np.arange(n_terminals)
width = 0.4
fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, y0, width, label=pi_names[0])
rects2 = ax.bar(x + width/2, y1, width, label=pi_names[1])

ax.set_ylabel('Sannolikhet')
ax.set_xlabel("Slutpoäng")
ax.set_title('Sannolikhet för slutpoäng, summering')
ax.legend()
fig.tight_layout()

plt.show()