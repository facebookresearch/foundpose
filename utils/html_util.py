#!/usr/bin/env python3

import base64

import cv2
from skimage.color import label2rgb


def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags
    """
    # img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", ndarray)
    return base64.b64encode(buffer).decode("utf-8")


def getHTMLImageBlob(encoded_str, image_extension):
    return (
        '<img width="200" src="data:image/'
        + image_extension
        + ";base64,"
        + encoded_str
        + '"/>'
        + " \n"
    )


def wrapHTMLBody(data):
    return "<html> <body> " + data + "</body>" + " \n" + "</html>"


def linebreakHTML(image_content):
    return image_content + "<p></p>" + " \n"


def writeHTML(outfile, image_content):
    html_str = wrapHTMLBody(image_content)
    with open(outfile, "w") as f:
        f.write(html_str)


def add_rgb(rgb_img, image_extension):
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    im_64 = ndarray_to_b64(rgb_img)
    html_blob = getHTMLImageBlob(im_64, image_extension)
    return html_blob


def add_depth(depth_img, image_extension):
    depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    im_64 = ndarray_to_b64(depth_img)
    html_blob = getHTMLImageBlob(im_64, image_extension)
    return html_blob


def add_labels(label_img, image_extension):
    # labels = label2rgb(label_img+1)
    labels = label2rgb(label_img)
    labels = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    im_64 = ndarray_to_b64(labels)
    html_blob = getHTMLImageBlob(im_64, image_extension)
    return html_blob


def add_text(text):
    return "<p> " + text + "</p>" + "\n"


def spitHTML(rgb_img, depth_img, label_gt, label_pred, metrics, img_extension):
    # Example usage of per-line image saver.

    image_content = ""
    image_content += add_text(img_extension, metrics)
    image_content += add_rgb(rgb_img, img_extension)
    image_content += add_depth(depth_img, img_extension)
    image_content += add_labels(label_gt, img_extension)
    image_content += add_labels(label_pred, img_extension)

    image_content = linebreakHTML(image_content)

    return image_content
