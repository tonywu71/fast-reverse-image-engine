##### This module is destined to import images and labels from our sql dataset #############
##### It will use the sqlite library in python to do the necessaries requests  #############

import sqlite3 as sql
import random
import numpy as np

from PIL import Image
import requests
from io import BytesIO as bio

media_l = ['media_comic', 'media_3d_graphics', 'media_vectorart', 'media_graphite', 'media_pen_ink', 'media_oilpaint', 'media_watercolor']
emotions_l = ['emotion_happy', 'emotion_scary', 'emotion_gloomy', 'emotion_peaceful']
content_l = ['content_building', 'content_flower', 'content_bicycle', 'content_people', 'content_dog', 'content_cars', 'content_cat', 'content_tree', 'content_bird']

def load_image(url = ''):
    try:
        reponse = requests.get(url)
        img = Image.open(bio(reponse.content))
        return img
    except:
        return load_image(url = 'https://rhyshonemusic.files.wordpress.com/2013/03/munch-scream-by-pastiche.jpg')

def show_image(img):
    img.show()
    return

def preprocess(x):
    return np.array(x)


class data_base:
    def __init__(self, file_path = 'data.sqlite'):
        self.file_path = file_path
        self.mids = []
        self.classes = {0: 'content_building', 1:'emotion_happy', 2:'content_flower',
                    3:'content_bicycle', 4:'media_comic', 5:'content_people',
                    6:'media_3d_graphics',7:'content_dog',8:'media_vectorart',9:'emotion_scary',
                    10:'emotion_gloomy', 11: 'media_graphite', 12: 'emotion_peaceful', 13: 'media_pen_ink',
                    14:'content_cars', 15:'media_oilpaint', 16: 'content_cat', 17: 'content_tree', 18: 'content_bird',19:'media_watercolor'}
        self.classes_labels = {self.classes[i]:i for i in self.classes}

        self.classes_emotions = {0:'emotion_happy',1:'emotion_scary',2:'emotion_gloom',3: 'emotion_peaceful'}

        self.classes_labels_emotions = {self.classes_emotions[i]:i for i in self.classes_emotions}

        self.classes_media = {0:'media_comic',1:'media_3d_graph',2:'media_vectorart',
        3: 'media_graphite',4: 'media_pen_ink' ,5:'media_oilpaint',6:'media_watercolor'}

        self.classes_labels_media = {self.classes_media[i]:i for i in self.classes_media}

    def get_image(self,mid): ## Gives the image identifier (MID) and returns the link to the image
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
                SELECT src FROM MODULES
                INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
                where MODULES.mid = ?
        ''',(mid,))
        result = cursor.fetchall()
        if(len(result) == 0):
            print("############# MID = ", mid," DOES NOT EXIST #########################")
            return result
        return result[0][0]

    def get_label(self,mid): ## Gives the image identifier (MID) and returns the label to the image
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
                SELECT *  FROM AUTOMATIC_LABELS WHERE MID = ?
        ''',(mid,))
        result = cursor.fetchall()
        result = list(result[0])
        labels = []
        #print(result)
        for i in result[1:]:
            if(i.lower() == 'positive' and not i.lower() == 'unsure' ):
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def get_label_labels(self,mid,label): ## Gives the image identifier (MID) and returns the link to the image
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        labels_aux = []
        labels = []
        for i in label:
            cursor.execute('''
                    SELECT "{}"  FROM AUTOMATIC_LABELS WHERE MID = ?
                    '''.format(i.replace('"', '""')),(mid,))
            result = cursor.fetchall()
            labels_aux.append(result[0][0])
            #print(result[0])
        #print(result)
        for i in labels_aux:
            if(i.lower() == 'positive' and not i.lower() == 'unsure' ):
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def return_classes(self,mid):
        labels = self.get_label(mid)
        classes = []
        for i in range(len(labels)):
            if(labels[i]==1):
                classes.append(self.classes[i])

        return classes

    def get_label_emotions(self,mid): ## Gives the image identifier (MID) and returns the link to the image
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
                SELECT MID, emotion_happy,emotion_scary,emotion_gloomy,emotion_peaceful  FROM AUTOMATIC_LABELS WHERE MID = ?
        ''',(mid,))
        result = cursor.fetchall()
        result = list(result[0])
        #print(result)
        labels = []
        #print(result)
        for i in result[1:]:
            if(i.lower() == 'positive' and not i.lower() == 'unsure' ):
                labels.append(1)
            else:
                labels.append(0)
        return labels

        def return_classes(self,mid):
            labels = self.get_label(mid)
            classes = []
            for i in range(len(labels)):
                if(labels[i]==1):
                    classes.append(self.classes[i])

            return classes

        def return_classes_emotions(self,mid):
            labels = self.get_label_emotions(mid)
            classes = []
            for i in range(len(labels)):
                if(labels[i]==1):
                    classes.append(self.classes_emotions[i])

            return classes




    def get_images_(self,n=100000 ):
        ### creating a cursor to the data base to use SQL commands

        ### Importing data
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
           SELECT MODULES.mid
           FROM MODULES
           INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
           limit ?
        ''', (n,))

        results = cursor.fetchall()
        images = {}
        random.shuffle(results)
        IDs = []
        for img in results:
            IDs.append(img[0])
            #print(img[0],"\n")
        self.mids = IDs
        return IDs

    def get_images(self,n=100000 ):
        ### creating a cursor to the data base to use SQL commands

        ### Importing data
        proportion = n//5;
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        IDs = []
        for i in self.classes_labels:


            cursor.execute('''
                SELECT MODULES.mid
                FROM MODULES
                INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
                WHERE AUTOMATIC_LABELS."{}" = 'positive'
                order by MODULES.mid
                limit ?
                '''.format(i.replace('"', '""')), (3*proportion,))

            results = cursor.fetchall()
            images = {}
            random.shuffle(results)
            j=0;
            for img in results:
                if(j>proportion):
                    break;
                if(img[0] not in IDs):
                    j=j+1;
                    IDs.append(img[0])
                    #print(img[0])
                #print(img[0],"\n")


        self.mids = IDs
        return IDs

    def get_images_emotions(self,n=100000 ):
        ### creating a cursor to the data base to use SQL commands

        ### Importing data
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
           SELECT MODULES.mid, AUTOMATIC_LABELS.emotion_happy, AUTOMATIC_LABELS.emotion_gloomy,  AUTOMATIC_LABELS.emotion_scary, AUTOMATIC_LABELS.emotion_peaceful
           FROM MODULES
           INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
           Where AUTOMATIC_LABELS.emotion_happy = 'positive' OR AUTOMATIC_LABELS.emotion_gloomy = 'positive' OR AUTOMATIC_LABELS.emotion_scary = 'positive'
           OR AUTOMATIC_LABELS.emotion_peaceful = 'positive'
           limit ?
        ''',(n,))

        results = cursor.fetchall()
        images = {}
        random.shuffle(results)
        IDs = []
        i = 0;
        for img in results:
            #label = self.get_label_emotions(img[0])
            #if(1 not in label):
            #    continue;
            IDs.append(img[0])
            #print(img)
            #print(img[0],"\n")
        self.mids = IDs
        return IDs

    def get_images_media(self,n=100000 ):
    ### creating a cursor to the data base to use SQL commands

    ### Importing data
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT MODULES.mid, AUTOMATIC_LABELS.media_comic AUTOMATIC_LABELS.media_3d_graph,  AUTOMATIC_LABELS.media_vectorart, AUTOMATIC_LABELS.media_graphite
        AUTOMATIC_LABELS.media_pen_ink AUTOMATIC_LABELS.media_oilpaint AUTOMATIC_LABELS.media_watercolor
        FROM MODULES
        INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
        Where AUTOMATIC_LABELS.media_comic = 'positive' OR AUTOMATIC_LABELS.media_3d_graph = 'positive' OR AUTOMATIC_LABELS.media_vectorart = 'positive'
        OR AUTOMATIC_LABELS.media_graphite= 'positive' OR AUTOMATIC_LABELS.media_pen_ink= 'positive' OR AUTOMATIC_LABELS.media_oilpaint= 'positive'
        OR AUTOMATIC_LABELS.media_watercolor= 'positive'
        limit ?
        ''',(n,))

        results = cursor.fetchall()
        images = {}
        random.shuffle(results)
        IDs = []
        i = 0;
        for img in results:
            #label = self.get_label_emotions(img[0])
            #if(1 not in label):
            #    continue;
            IDs.append(img[0])
            #print(img)
            #print(img[0],"\n")
        self.mids = IDs
        return IDs

    def get_images_labels(self,n=10000,labels=[]):
        ### creating a cursor to the data base to use SQL commands

        ### Importing data
        proportion = n//len(labels);
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        IDs = []
        for i in labels:


            cursor.execute('''
                SELECT MODULES.mid
                FROM MODULES
                INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
                WHERE AUTOMATIC_LABELS."{}" = 'positive'
                order by MODULES.mid
                limit ?
                '''.format(i.replace('"', '""')), (5*proportion,))

            results = cursor.fetchall()
            images = {}
            random.shuffle(results)
            j=0;
            for img in results:
                if(j>proportion):
                    break;
                if(img[0] not in IDs):
                    j=j+1;
                    IDs.append(img[0])
                    #print(img[0])
                #print(img[0],"\n")


        self.mids = IDs
        return IDs

    def get_label_media(self,mid): ## Gives the image identifier (MID) and returns the link to the image
        conn = sql.connect(self.file_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MID, media_comic, media_3d_graph, media_vectorart,
            media_graphite, media_pen_ink, media_oilpaint, media_watercolor WHERE MID = ?
            ''',(mid,))
        result = cursor.fetchall()
        result = list(result[0])
        #print(result)
        labels = []
        #print(result)
        for i in result[1:]:
            if(i.lower() == 'positive' and not i.lower() == 'unsure' ):
                labels.append(1)
            else:
                labels.append(0)

        return labels

    def train_test_split(self,n = 1000,test_frac = 0.3, preprocess = preprocess):
        if(len(self.mids) == 0):
            self.get_images(n)
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for i in range(int((1-test_frac)*n)):
            mid = self.mids[i][0]
            url = self.get_image(mid)
            x = load_image(url)
            x = preprocess(x)
            train_x.append(x)
            train_y.append(self.get_label(mid))

        #print(train_x)
        for i in range(int((1-test_frac)*n),n):
            mid = self.mids[i][0]
            url = self.get_image(mid)
            x = load_image(url)
            x = preprocess(x)
            test_x.append(x)
            test_y.append(self.get_label(mid))

        return train_x, train_y, test_x, test_y



def tests():
    db = data_base()
    db.get_images(10000)
    #print(db.get_label('489'))
    #print(db.return_classes('489'))
    #x,y,z,w = db.train_test_split(10)
    #print(x)

#tests()
