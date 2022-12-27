import numpy as np
import csv
import time

from sklearn import svm
import pandas as pd

#Database: Gerbang LOgika AND
#Membaca data dari file
FileDB = 'DatabaseGerakParabola.txt'
Database = pd.read_csv(FileDB, sep=",", header=0)

#x = Data, y=Target
X = Database[[u't']]
y = [Database.Posisiy, Database.Posisix]


clf = svm.SVC()
clf.fit(X.values,y)

print(clf.predict( [[0.2]] ))
print(clf.predict( [[0.3]] ))
print(clf.predict( [[1.5]] ))
print(clf.predict( [[2.2]] ))
print(clf.predict( [[2.5]] ))

