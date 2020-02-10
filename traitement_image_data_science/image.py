import matplotlib.pyplot as plt # afin d'afficher l'image
import numpy as np  # sert �  utiliser des matrices
import pickle   # sert �  importer l'image de CIFAR-10

# transforme l'ensemble des images en dictionnaire
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# retourne l'image i de dict
def image(index_image, file):
    picture = np.zeros((32, 32, 3), dtype=np.uint8)
    dict = unpickle(file)
    # parcour des trois couleurs diff�rentes
    for i in range(3):
        # parcour des des colonnes
        for j in range(1024*i, 1024*(i+1)):
            lig = (j-1024*i) // 32
            col = (j-1024*i) % 32
            picture[lig][col][i] = dict[data][index_image][i]
    return picture

def affiche_image(index_image, file):
    plt.imshow(image(index_image, file))
    plt.show()