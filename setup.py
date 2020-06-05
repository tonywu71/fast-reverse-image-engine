import os
import requests


print('Downloading JSON Model')
try:
    model = requests.get('https://docs.google.com/uc?export=download&confirm=y6Pt&id=1FhWXbVpe8xtkM15PvC7FSFXQJGvdGnfb')
    f = open('feature_extractor/models/feature_extractor_90_256.json','wb')
    f.write(model.content)
    f.close()
except:
    print('Failure in downloading the JSON model')



print("Downloading weights of the Model - This can take a While")
print("###################### This Can take a while ####################################")
try:
    model = requests.get('https://download1075.mediafire.com/2au7l9ij6zog/l7994qxgew14smr/feature_extractor_90_256.h5', verify = False)
    f = open('feature_extractor/models/feature_extractor_90_256.h5','wb')
    f.write(model.content)
    f.close()
except:
    print("Failure in downloading the model weights")

print("Downloading the LSH - This can take a While")
print("###################### This Can take a while ####################################")
try:
    model = requests.get('https://download1075.mediafire.com/2au7l9ij6zog/l7994qxgew14smr/feature_extractor_90_256.h5', verify = False)
    f = open('LSHash/lsh_256_100.p','wb')
    f.write(model.content)
    f.close()
except:
    print("Failure in downloading the model LSHash weights")

answer = input("Do you want to download the database as well ? (OBS: It is not necessary to just deploy the web server ? (y or n) : ")
if(answer.lower()[0]=='y'):

    print('Downloading DataBase - This can take a while')
    try:
        database = requests.get("https://download1084.mediafire.com/ainc0wyevhxg/34acq1ylvxr8497/data.sqlite",verify= False)
        f = open('data/data.sqlite','wb')
        f.write(database.content)
        f.close()
    except:
        print('Failure in downloading the database')
