# from https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil

import PIL.Image as Image
import numpy as np
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img


if __name__ == "__main__":
    img_path = "/workspace/Data/SAPIEN/laptop/mv_laptop_500_4_IF/val/manual/frame_00000000_view_00_color00.png"
    img = Image.open(img_path)
    img.save("test_origin.png")
    hue = 20
    img = colorize(img, hue)
    img.save("test_hue{}.png".format(hue))
