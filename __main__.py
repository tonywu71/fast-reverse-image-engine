import sys
sys.path.append('data/')
sys.path.append('feature_extractor/')
sys.path.append('LSHash/')

tags = ['-l','-p','--help']

if(len(sys.argv) < 2):
    print("Please provide a link(-l)/path(-p) to a image using the arguments -l or -p,  if not clear type:");
    print("'python3 __main__.py --help'");

elif(sys.argv[1] == '--help' or sys.argv[1] not in tags):
    print("\n################# How to use ##########################")
    print("\nTo search a image from a link use the tag -l, \n ex: 'python3 __main__.py -l example.com/img.jpg'\n")
    print("\nTo search a image from a path use the tag -p, \n ex: 'python3 __main__.py -p mypath/img.jpg'\n")
    print("\nTo see the ways of how to use please use the tag --help,\n ex: 'python3 __main__.py --help'\n")

else:
    from feature_extractor import *
    from Data_Gen import *
    from PIL import Image
    from lshash import LSHash
    import pickle
    fe = feature_extractor(model_path='feature_extractor/models/feature_extractor_90.json',weights_path='feature_extractor/models/feature_extractor_90.h5')

def get_similar_item(img, lsh_variable, n_items=5,url = False):
    if(url):
        img = load_image(img)
        img = preprocess_image(img)
        img = np.expand_dims(img,axis = 0)

    vect = fe.predict(img)[0]
    print(vect)
    response = lsh_variable.query(vect,
                     num_results=n_items+1, distance_func='euclidean')

    return response

def main():
    lsh = pickle.load(open('LSHash/lsh.p', "rb"))

    if(sys.argv[1]       == '-l'):
        res = get_similar_item(sys.argv[2],lsh,url=True)

    elif(sys.argv[1] == '-p'):
        res = get_similar_item(sys.argv[2],lsh,url=False)

    for i in res:
        im = load_image(i[0][1])
        im.show()

main()
