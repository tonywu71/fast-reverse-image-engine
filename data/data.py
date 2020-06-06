##### This module is destined to import images and labels from our sql dataset #############
##### It will use the sqlite library in python to do the necessaries requests  #############

import sqlite3 as sql
import random
import numpy as np

from PIL import Image
import requests
from io import BytesIO as bio

# CHANGE NEEDED: Please Matheus try to comment using '''...''' w/ inputs and outputs for all the function that are useful.

# Below, the different categories of labels are grouped together in lists.
media_l = ['media_comic', 'media_3d_graphics', 'media_vectorart', 'media_graphite', 'media_pen_ink', 'media_oilpaint', 'media_watercolor']
emotions_l = ['emotion_happy', 'emotion_scary', 'emotion_gloomy', 'emotion_peaceful']
content_l = ['content_building', 'content_flower', 'content_bicycle', 'content_people', 'content_dog', 'content_cars', 'content_cat', 'content_tree', 'content_bird']

def load_image(url = ''):
    # FIX NEEDED: wrong behaviour with the except, maybe just raise an error?
    # FEATURE NEEDED: maybe in another function, but create a function that detects if the image is inexistent,
    # so it'll make easier for Lucie to clean up the data base.
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
    # I don't quite understand why you have to convert x to an array, maybe you could explain it quickly?
    return np.array(x)


class data_base:
    # COMMENT NEEDED: Just a quick description of the class
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
        # HEADER NEEDED '''...'''
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

    def get_label(self,mid): ## Gives the image identifier (MID) and returns the link to the image
        # HEADER NEEDED '''...'''
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
        # HEADER NEEDED '''...'''
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


    def return_classes(self,mid): # return the classes that a image belongs

        labels = self.get_label(mid)
        classes = []
        for i in range(len(labels)):
            if(labels[i]==1):
                classes.append(self.classes[i])

        return classes

    def get_label_emotions(self,mid): ## Gives the image identifier (MID) and returns the link to the image
        # Also we've talked about it the 4th of May but maybe just replace the code with return get_label_labels(self,mid,label=emotions_l)
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





    def get_images_(self,n=100000 ): # return the n images
        ### creating a cursor to the data base to use SQL commands
        # HEADER NEEDED '''...'''

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

    def get_images(self,n=100000 ): # return n images but with good proportion between classes
        ### creating a cursor to the data base to use SQL commands
        # HEADER NEEDED '''...'''

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



    def get_images_labels(self,n=10000,labels=[]): # return the n images that belongs to the classes that are in labels
        ### creating a cursor to the data base to use SQL commands
        # HEADER NEEDED '''...'''

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


def tests():
    db = data_base()
    db.get_images(100)
    #print(db.get_label('489'))
    #print(db.return_classes('489'))
    #x,y,z,w = db.train_test_split(10)
    #print(x)

#tests()
