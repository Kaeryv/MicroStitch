import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def inverse_colormap(rgb, cmap="hsv"):
    color_map = plt.cm.get_cmap(cmap)
    master = color_map(np.linspace(0,1, 180))[:,:3]
    norms = list()
    for m in master:
        norm = np.sqrt(np.sum(np.abs(m - rgb)**2))
        norms.append(norm)
    return 180 * master[np.argmin(norms)]

def img_inv_cmap(rgbmap, cmap="hsv"):
    for i in trange(rgbmap.shape[0]):
        for j in range(rgbmap.shape[1]):
            rgbmap[i, j] = inverse_colormap(rgbmap[i, j])


if __name__ == "__main__":
    color_map = plt.cm.get_cmap('hsv')
    x = np.linspace(0,1, 100)
    print(color_map(x))

    colors = color_map(x)
    plt.scatter(x, x, c = color_map(x))
    #plt.plot([inverse_colormap(xx) for xx in x])

    plt.show()