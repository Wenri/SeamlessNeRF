import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from Additive_mixing_layers_extraction import Hull_Simplification_determined_version, \
    Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates, \
    recover_ASAP_weights_using_scipy_delaunay


def save_weights(img, palette_rgb, mixing_weights, output_prefix):
    mixing_weights = mixing_weights.reshape((img.shape[0], img.shape[1], -1)).clip(0, 1)
    temp = (mixing_weights.reshape((img.shape[0], img.shape[1], -1, 1)) * palette_rgb.reshape((1, 1, -1, 3))
            ).sum(axis=2)
    img_diff = temp * 255 - img * 255
    diff = np.square(img_diff.reshape((-1, 3))).sum(axis=-1)
    print('max diff: ', np.sqrt(diff).max())
    print('median diff', np.median(np.sqrt(diff)))
    rmse = np.sqrt(diff.sum() / diff.shape[0])
    print('RMSE: ', np.sqrt(diff.sum() / diff.shape[0]))

    import json
    mixing_weights_filename = output_prefix + "-palette_size-" + str(len(palette_rgb)) + "-mixing_weights.js"
    with open(mixing_weights_filename, 'w') as myfile:
        json.dump({'weights': mixing_weights.tolist()}, myfile)

    for i in range(mixing_weights.shape[-1]):
        mixing_weights_map_filename = output_prefix + "-palette_size-" + str(
            len(palette_rgb)) + "-mixing_weights-%02d.png" % i
        Image.fromarray((mixing_weights[:, :, i] * 255).round().clip(0, 255).astype(np.uint8)).save(
            mixing_weights_map_filename)
    return rmse


def get_bigger_palette_to_show(palette):
    ##### palette shape is M*3
    c = 50
    palette2 = np.ones((1 * c, len(palette) * c, 3))
    for i in range(len(palette)):
        palette2[:, i * c:i * c + c, :] = palette[i, :].reshape((1, 1, -1))
    return palette2


def print_hi(filepath: str | os.PathLike[str]):
    filepath = Path(filepath)
    img = np.asfarray(Image.open(filepath).convert('RGB')) / 255.0
    arr = img.copy()
    X, Y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    XY = np.dstack((X * 1.0 / img.shape[0], Y * 1.0 / img.shape[1]))
    data = np.dstack((img, XY))
    print(len(data.reshape((-1, 5))))

    start = time.time()
    palette_rgb = Hull_Simplification_determined_version(
        img, filepath.stem + "-convexhull_vertices", error_thres=1. / 256.)
    end = time.time()
    M = len(palette_rgb)
    print("palette size: ", M)
    print("palette extraction time: ", end - start)

    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img * 255).round().astype(np.uint8)).save(filepath.stem + "-convexhull_vertices.png")

    ######### for RGBXY RGB black star triangulation.
    start = time.time()
    data_hull = ConvexHull(data.reshape((-1, 5)))
    start2 = time.time()
    print("convexhull on 5D time: ", start2 - start)
    mixing_weights_1 = Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(
        img.reshape((-1, 3))[data_hull.vertices].reshape((-1, 1, 3)), palette_rgb, "None", order=0, SAVE=False)
    mixing_weights_2 = recover_ASAP_weights_using_scipy_delaunay(
        data_hull.points[data_hull.vertices], data_hull.points, option=3)

    mixing_weights = mixing_weights_2.dot(mixing_weights_1.reshape((-1, M)))

    end = time.time()
    print("total time: ", end - start)

    mixing_weights = mixing_weights.reshape((img.shape[0], img.shape[1], -1)).clip(0, 1)

    output_prefix = filepath.stem + '-RGBXY_RGB_black_star_ASAP'
    return save_weights(arr, palette_rgb, mixing_weights, output_prefix)


if __name__ == '__main__':
    print_hi('../nerf-pytorch/data/nerf_llff_data/fern/images/IMG_4026.JPG')
