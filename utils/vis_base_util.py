#!/usr/bin/env python3

"""2D visualization of primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.

Ref: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz.py
"""

from typing import Optional, Sized, Tuple, List
import os 
import cv2
import matplotlib
import matplotlib.cm
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('agg') # since MacOSX backend uses different config. 

def normalize_data(img):
    return (img - img.min()) / (img.max() - img.min())


def get_colormap(num_colors, cmap=cv2.COLORMAP_TURBO):

    palette = np.linspace(0, 255, num_colors, dtype=np.uint8).reshape(1, num_colors, 1)
    palette = cv2.applyColorMap(palette, cmap)
    palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB).squeeze(0)
    palette = normalize_data(palette)
    return palette.tolist()


def cm_RdGn(x: int):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(
    imgs,
    titles=None,
    cmaps="gray",
    dpi: int = 100,
    pad: float = 0.0,
    adaptive: bool = True,
) -> None:
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    im_width = imgs[0].shape[1]
    im_height = imgs[0].shape[0]

    # Check that all images have the same size.
    if n > 1:
        for im_id in range(len(imgs[1:])):
            if imgs[im_id].shape[1] != im_width or imgs[im_id].shape[0] != im_height:
                raise ValueError("Images must be of the same size.")

    figsize = (len(imgs) * im_width / dpi, im_height / dpi)
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi) 

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4) -> None:
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(ps, list):
        ps = [ps] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c, p in zip(axes, kpts, colors, ps):
        a.scatter(k[:, 0], k[:, 1], c=c, s=p, linewidths=0)


def plot_mspd_keypoints(
    mspd_id, projs_gt, projs_est, image_width, image_height, ps: int = 2
) -> None:
    """
        Plots the mspd error gt keypoint and the projection keypoint.
        Assumes the image is already on the right side of the current plot.
    Args:
        mspd_id: vertice id returned by the error calculator.
        projs_gt: vertices projected in gt pose.
        projs_est: vertices_projected in est pose.

    """
    fig = plt.gcf()
    # ax = fig.axes[0]
    ax = plt.gca()

    gt_keypoint = projs_gt[mspd_id]
    est_keypoint = projs_est[mspd_id]
    # clip estimated keypoint to fit within the image:
    gt_keypoint[0] = max(gt_keypoint[0], 0)
    gt_keypoint[0] = min(gt_keypoint[0], image_width)
    gt_keypoint[1] = max(gt_keypoint[1], 0)
    gt_keypoint[1] = min(gt_keypoint[1], image_height)

    est_keypoint[0] = max(est_keypoint[0], 0)
    est_keypoint[0] = min(est_keypoint[0], image_width)
    est_keypoint[1] = max(est_keypoint[1], 0)
    est_keypoint[1] = min(est_keypoint[1], image_height)

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax.transData.transform(gt_keypoint))
    fkpts1 = transFigure.transform(ax.transData.transform(est_keypoint))
    fig.lines += [
        matplotlib.lines.Line2D(
            (fkpts0[0], fkpts1[0]),
            (fkpts0[1], fkpts1[1]),
            zorder=1,
            transform=fig.transFigure,
            c="red",
            linewidth=1.0,
            alpha=1.0,
        )
    ]

    ax.scatter(gt_keypoint[0], gt_keypoint[1], c="green", s=ps)
    ax.scatter(est_keypoint[0], est_keypoint[1], c="red", s=ps)


def plot_matches(
    kpts0,
    kpts1,
    color: List[int] = None,
    lw: float = 1.5,
    ps: int = 4,
    indices: Tuple[int, int] = (0, 1),
    a=1.0,
    w: int = 640,
    h: int = 480,
) -> None:
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: (int or list) alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    # filter out out of image keypoints on query (kpts1)
    mask = np.logical_and(
        np.logical_and(0 <= kpts1[:, 0], kpts1[:, 0] < w),
        np.logical_and(0 <= kpts1[:, 1], kpts1[:, 1] < h),
    )
    kpts1 = kpts1[mask]
    kpts0 = kpts0[mask]

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)
    else:
        color = np.array(color)[mask].tolist()

    if type(a) != list:
        a = [a for i in range(len(kpts0))]

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=color[i],
                linewidth=lw,
                alpha=a[i],
            )
            for i in range(len(kpts0))
        ]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_boundingbox(box) -> None:
    """Plot rectangle to show object bounding box in the image.
    Args:
        box: ndarrays of size [4].
    """
    x1, y1, x2, y2 = box
    crop_width, crop_height = x2 - x1, y2 - y1
    rect = patches.Rectangle(
        (x1, y1),
        crop_width,
        crop_height,
        linewidth=1,
        edgecolor="white",
        facecolor="none",
    )
    # Add the patch to the Axes
    ax = plt.gca()
    ax.add_patch(rect)


def plot_losses(losses) -> None:
    """Plot losses."""
    plt.figure()
    plt.title("Training losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(losses)


def plot_curve(
    x, y, xlabel: str = "x axis", ylabel: str = "y axis", title: str = "plot"
) -> None:
    """Plot losses."""
    plt.figure()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(x, y)

def plot_boxplot(values, title: str = "brightness") -> None:

    fig, ax = plt.subplots()

    # boxplot for average brightness values across all images
    ax.boxplot(values, vert=True, patch_artist=True, labels=["Brightness"])

    # display variance of values in the title
    ax.set_title(f"Box plot of {title}\nVariance: {np.var(values):.4f}")

def plot_histogram(
    hist_values,
    n_bins,
    value,
    im_width: int = 630,
    im_height: int = 476,
    dpi: int = 100,
    pad: float = 0.0,
    colors: List[int] = None,
    amb: bool = False,
) -> None:
    """Plot histogram of values."""

    figsize = (im_width / dpi, im_height / dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    n, bins, patches = ax.hist(hist_values, bins=n_bins)
    ax.set_xlabel(value)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {value}")

    if colors is not None:
        if len(colors) == len(bins):
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[i])
        elif amb:
            prev = 0
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[prev])
                if n[i] != 0:
                    prev += 1
        else:
            prev = 0
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[prev])
                if n[i] != 0:
                    prev += int(n[i])

    # plt.rcParams["font.family"] = "Arial"

    fig.tight_layout(pad=pad)


def plot_bar(
    features,
    importances,
    indices,
    x_label: str = "Relative Importance",
    title: str = "Feature importances using MDI",
    figsize: Tuple[int, int] = (20, 20),
) -> None:
    """Plot bar chart of mean and standard deviation"""
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=7)
    plt.xlabel(x_label)

def plot_tsne(x, y) -> None:
    """
        Creates a scatter plot for tsne visualization.
    Args:
    x: input data of shape (N, 2).
    y: output data of shape (N).
    """
    class_labels = y  # np.random.randint(0, np.unique(y).list(), y.shape[0])

    # Create a color map for the classes (using 'tab20b' colormap from Matplotlib)
    cmap = plt.get_cmap("tab20b")
    num_classes = len(np.unique(class_labels))
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        mask = class_labels == i
        plt.scatter(
            x[mask, 0], x[mask, 1], label=f"Class {i}", color=colors[i], alpha=0.7
        )

    plt.title("t-SNE Visualization of Vertices")

def add_contour_overlay(
    img,
    render_img,
    color: Optional[Tuple] = (255, 255, 255),
    dilate_iterations: Optional[int] = 1,
):
    """
    Overlays object boundaries on a given imaged.
    Boundaries are estimated from an object rendered image.

    Ref: https://github.com/megapose6d/megapose6d/blob/master/src/megapose/visualization/utils.py#L47
    """

    img_t = torch.as_tensor(render_img)
    mask = torch.zeros_like(img_t)
    mask[img_t > 0] = 255
    mask = torch.max(mask, dim=-1)[0]
    mask_bool = mask.numpy().astype(np.bool_)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
    mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    img_contour = np.copy(img)
    img_contour[canny > 0] = color

    return img_contour


def add_text(
    idx,
    text,
    pos: Tuple[float, float] = (0.01, 0.99),
    fs: int = 12,  # 15,
    color: str = "w",
    lcolor: str = "k",
    lwidth: int = 2,
    ha: str = "left",
    va: str = "top",
) -> None:

    fig = plt.gcf()
    ax = fig.axes[idx]

    zorder = ax.get_zorder()
    if len(fig.axes) > 1:
        zorder = max(zorder, fig.axes[-1].get_zorder())
        fig.axes[-1].set_axisbelow(True)

    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
        zorder=zorder + 5,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )
        t.set


def save_plot(path, **kw) -> None:
    """Save the current figure without any white margin."""

    with os.path.open(path, "wb") as f:
        plt.savefig(f, bbox_inches="tight", pad_inches=0, **kw)

def save_plot_to_ndarray():

    fig = plt.gcf()
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.clf()
    plt.cla()
    plt.close(fig)

    return data
