import sqlite3 as sql
from data import *
import requests
from io import BytesIO as bio
from PIL import Image

import multiprocessing as mp

def get_url_i(i,cursor):
    cursor.execute('''
           SELECT MODULES.src
           FROM MODULES
           INNER JOIN AUTOMATIC_LABELS on MODULES.mid = AUTOMATIC_LABELS.mid
           limit ? offset ?
        ''', (1,i))

    results = cursor.fetchall()
    return results[0][0]

def clean(init,end,q):
    print('ok')
    conn = sql.connect('data/data.sqlite')
    cursor = conn.cursor()
    i = init;
    n= end
    i_rad = []
    while(i<n):
        #print('ok')
        results = get_url_i(i,cursor)
        try:
            reponse = requests.get(results)
            img = Image.open(bio(reponse.content))
        except:
            i_rad.append(i)
        reponse = requests.get(results)
        img = Image.open(bio(reponse.content))
        i=i+1;
        if(i%50== 0):
            print(i)
    i_rad.append('ok')
    q.put(i_rad)
    return i_rad;


def main():
    conn = sql.connect('data/data.sqlite')
    cursor = conn.cursor()
    processes = []
    queues = []
    i_errado = []
    n = 2
    N = 20
    part = N//n
    for i in  range(n):
        q =  mp.Queue()
        #print(part*(i+1) - i*part)
        p = mp.Process(target=clean, args=(i*part,part*(i+1),q))
        p.start()
        processes.append(p)
        queues.append(q)
        print('started ' + str(i))

    for i in range(n):
        processes[i].join()
        i_errado = queues[i].get() + i_errado

    print(i_errado)



if __name__ == '__main__':
    main()
