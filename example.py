import csv
import numpy as np
from sklearn import metrics



user2item = {}
item2user = {}

###########################
# read the train ratings
f = open('./train_ratings.csv')
lines = np.array((f.readlines())[1:])

for idx, l in enumerate(lines):
    # print(l)
    try:
        user,item,rating = l.split(',')
    except:
        user, item, rating = l.split('"')
        user = user[:-1]
        rating = rating[1:]
    if(user not in user2item.keys()):
        user2item[user]={}
    if(item not in item2user.keys()):
        item2user[item]={}

    user2item[user][item]=1
    item2user[item][user]=1

###########################
# Build a model

###########################
# Make prediction
f = open('./test_ratings.csv')
lines = np.array((f.readlines())[1:])

pred_total = []
gnd_total = []

for idx, l in enumerate(lines):
    # print(l)
    try:
        user,item,rating = l.split(',')
    except:
        user, item, rating = l.split('"')
        user = user[:-1]
        rating = rating[1:]
    
    predicted_rating = 5
    
    gnd_total.append(rating)
    pred_total.append(predicted_rating)


RMSE = metrics.mean_squared_error(gnd_total, pred_total, squared = False)

print("RMSE : {}".format(RMSE))
