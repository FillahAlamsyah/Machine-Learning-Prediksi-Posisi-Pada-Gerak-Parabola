import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt

# Membaca data dari file
FileDB = r"C:\Users\fedba\Downloads\DatabaseGerakParabola.txt"
Database = pd.read_csv(FileDB, sep=",")

# Memisahkan data dan target
X = Database[['t']]
y = Database[['Posisix', 'Posisiy']]

# Memberikan nama fitur pada X
X.columns = ['t']

# Membuat model SVR untuk Posisix
clf_posisix = svm.SVR()
clf_posisix.fit(X, y['Posisix'])

# Membuat model SVR untuk Posisiy
clf_posisiy = svm.SVR()
clf_posisiy.fit(X, y['Posisiy'])

# Membuat prediksi untuk beberapa nilai waktu tertentu
time_values = np.linspace(0, 2.5, 10).reshape(-1, 1)

# Prediksi Posisix dan Posisiy
predictions_posisix = clf_posisix.predict(time_values)
predictions_posisiy = clf_posisiy.predict(time_values)

# Membuat subplot dengan 2 kolom dan 2 baris
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Plot Posisix
axs[0, 0].scatter(X, y['Posisix'], color='black', label='Actual Posisix')
axs[0, 0].plot(time_values, predictions_posisix, color='red', label='Predicted Posisix')
axs[0, 0].set_ylabel('Posisix')
axs[0, 0].legend()

# Plot Posisiy
axs[0, 1].scatter(X, y['Posisiy'], color='blue', label='Actual Posisiy')
axs[0, 1].plot(time_values, predictions_posisiy, color='green', label='Predicted Posisiy')
axs[0, 1].set_ylabel('Posisiy')
axs[0, 1].legend()

# Plot antara Posisix dan Posisiy
axs[1, 0].scatter(y['Posisix'], y['Posisiy'], color='purple', label='Actual Posisix vs Posisiy')
axs[1, 0].set_xlabel('Posisix')
axs[1, 0].set_ylabel('Posisiy')
axs[1, 0].legend()

# Plot antara predictions_posisix dan predictions_posisiy
axs[1, 1].scatter(predictions_posisix, predictions_posisiy, color='orange', label='Predicted Posisix vs Posisiy')
axs[1, 1].set_xlabel('Predicted Posisix')
axs[1, 1].set_ylabel('Predicted Posisiy')
axs[1, 1].legend()

# Menampilkan hasil prediksi
for time, posisix, posisiy in zip(time_values, predictions_posisix, predictions_posisiy):
    print(f"t = {time[0]:.2f} s, Prediksi (x,y) = ({posisix:.2f}, {posisiy:.2f}) m")

plt.show()
