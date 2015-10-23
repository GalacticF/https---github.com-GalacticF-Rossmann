import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import cross_validation 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
import csv


DOBIG = True

def ExtractData(df, IsTrain):
    df.replace('0', 0, inplace=True)
    df.replace('1', 1, inplace=True)
    df.replace('a', 2, inplace=True)
    df.replace('b', 3, inplace=True)
    df.replace('c', 4, inplace=True)
    df.replace('d', 5, inplace=True)
    #data = df.ix[:, df.columns != "Sales"]
    if (IsTrain):
        data = df[["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Store"]]
    else: 
        data = df[["DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Id"]]
    Yeras = pd.DataFrame(df["Date"].str.split('-').tolist(), columns = ['Y', 'M', 'D'])
    data['Y'] = Yeras['Y']
    data['M'] = Yeras['M']
    data['D'] = Yeras['D']    
    return data

def ConsolidateStoresData (storesdf, datadf):
    if (IsTrain):
        data = pd.merge()
    else: 


stores = pd.read_csv(r"E:\kaggle\STORES\store.csv", low_memory = False)
df = pd.read_csv(r"E:\kaggle\STORES\train.csv", low_memory = False)

ConsolidateStoresData(stores, df)
data = ExtractData(df, True)
labels = df["Sales"]
np_data = data.as_matrix()
np_labels = labels.as_matrix()
if DOBIG == False:
    train, test, labels_train, labels_test = cross_validation.train_test_split(np_data, np_labels, test_size = 0.2)


dftest = pd.read_csv(r"E:\kaggle\STORES\test.csv")

if DOBIG:
    dftest = dftest.fillna(0)
    df_test = ExtractData(dftest, False)

    #df_labels_test = dftest[["Sales"]]
    df_ids = dftest['Id']
    test_ids = df_ids.as_matrix()
    test = df_test.as_matrix()
    #labels_test = df_labels_test.as_matrix()
    train = np_data
    labels_train = np_labels

#clf = LinearRegression()
clf = SVR()
train = train.astype(int)
clf.fit(train, labels_train)
test = test.astype(int)
prd = clf.predict(test)

myfile = open("E:\kaggle\STORES\out.csv", 'wb')
wr = csv.writer(myfile)
out_file = []
out_file.append(["Id", "Sales"])
for i in xrange(0, len(prd)):
    out_file.append([int(test_ids[i]), int(prd[i])])
for e in out_file:
    wr.writerow(e)
#print accuracy_score(labels_test, prd)
#dftst = pd.read_csv(r"E:\kaggle\STORES\train.csv", low_memory = False)