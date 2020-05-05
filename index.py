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

fe = feature_extractor(model_path='feature_extractor/models/feature_extractor_90_256.json',weights_path='feature_extractor/models/feature_extractor_90_256.h5')

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



lsh = pickle.load(open('LSHash/lsh_256_100.p', "rb"))

options = ['Link', 'Upload Image']

class basicRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('web/index.html')

class linkRequestHandler(tornado.web.RequestHandler):
    def get(self):
        #print(links)
        self.write(json.dumps(options))

    def post(self):
        link_image = self.get_arguments("link")[0]
        print(link_image)
        links = []
        #links.append(link_image)
        res = get_similar_item(link_image,lsh, url = True)
        #print(res)
        for i in res:
            links.append(i[0][1])

        #print(links)
        self.write(json.dumps(links))

class uploadRequestHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["fileImage"]
        for f in files:
            if(f == None):
                self.render('index.html')
            img = Image.open(bio(f.body))
            #img.show()
            img = preprocess_image(img)
            img = np.expand_dims(img,axis = 0)

            links = []

            res = get_similar_item(img,lsh, url = False)

            for i in res:
                links.append(i[0][1])

        print(links)
        self.write(json.dumps(links))


    def get(self):
        self.render("index.html")



if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", basicRequestHandler),
        ("/list",linkRequestHandler ),
        ("/upload",uploadRequestHandler)
    ])

    app.listen(80)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()
