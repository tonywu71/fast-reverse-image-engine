import sys
sys.path.append('data/')
sys.path.append('feature_extractor/')
sys.path.append('LSHash/')
from feature_extractor import *
from Data_Gen import *
from PIL import Image
from lshash import LSHash
import pickle

import tornado.web
import tornado.ioloop
import json

fe = feature_extractor(model_path='feature_extractor/models/feature_extractor_90.json',weights_path='feature_extractor/models/feature_extractor_90.h5')

def get_similar_item(img, lsh_variable, n_items=5,url = False):
    if(url):
        img = load_image(img)
        img = preprocess_image(img)
        img = np.expand_dims(img,axis = 0)

    vect = fe.predict(img)[0]
    #print(vect)
    response = lsh_variable.query(vect,
                     num_results=n_items+1, distance_func='euclidean')

    return response



lsh = pickle.load(open('LSHash/lsh.p', "rb"))

links = ['teste','teste2']

class basicRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('web/index.html')

class linkRequestHandler(tornado.web.RequestHandler):
    def get(self):
        #print(links)
        self.write(json.dumps(links))

    def post(self):
        link_image = self.get_arguments("fruit")[0]
        print(link_image)
        links = []
        #links.append(link_image)
        res = get_similar_item(link_image,lsh, url = True)
        print(res)
        for i in res:
            links.append(i[0][1])

        print(links)
        self.write(json.dumps(links))


if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", basicRequestHandler),
        ("/list",linkRequestHandler )
    ])

    app.listen(8080)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()
