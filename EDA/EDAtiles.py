import pickle as pkl
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from shapely.wkt import loads


def show_tiles():
    '''
    Displays clipped images and its corresponding overlay tiles
    '''

    img_path = '/home/dlahtou/6040_2_2/imgs'
    shapes_path = '/home/dlahtou/6040_2_2/shapes'
    for i in range(169):
        with open(img_path+f'/image{i}', 'rb') as open_file:
            image = pkl.load(open_file)
        with open(shapes_path+f'/wkt{i}.pkl', 'rb') as open_file:
            shape = pkl.load(open_file)

        fig = plt.figure()
        for i in range(3):
            fig.add_subplot(1, 4, i+1)
            plt.imshow(image[i, :, :])

        clipped_poly = loads(shape)

        fig, ax = plt.subplots(figsize=[8, 8])
        for polygon in clipped_poly:
            mpl_poly = Polygon(np.array(polygon.exterior))
            ax.add_patch(mpl_poly)

        ax.relim()
        ax.autoscale_view()

        plt.show()
