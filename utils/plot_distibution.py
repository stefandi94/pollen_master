import numpy as np
import matplotlib.pyplot as plt

distros = {'Acer': 1176,
           'Alnus': 2128,
           'Artemisia': 13456,
           'Betula': 9937,
           'Cedrus': 842,
           'Corylus': 14251,
           'Cynodon': 1056,
           'Dactilis': 2165,
           'Fraxinus': 4694,
           'Picea': 2392,
           'Populus': 6505,
           'Quercus': 1427,
           'Salix': 9463,
           'Taxus': 5777,
           }

N = len(distros.keys())
data = distros.values()
ticks = distros.keys()
if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(ticks, data)
    x = np.arange(N)
    plt.xticks(x, ticks, rotation=45)
    # plt.show()

    plt.savefig("pollen_distribution" + ".png")
