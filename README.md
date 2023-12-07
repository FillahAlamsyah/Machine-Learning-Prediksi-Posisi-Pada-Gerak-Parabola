# Machine Learning : Prediksi Posisi Pada Gerak Parabola

1. Program Membuat Database : [GerakParabola.py](https://github.com/FillahAlamsyah/Machine-Learning-Prediksi-Posisi-Pada-Gerak-Parabola/blob/main/GerakParabola.py)
2. File Database : [DatabaseGerakParabola.txt](https://github.com/FillahAlamsyah/Machine-Learning-Prediksi-Posisi-Pada-Gerak-Parabola/blob/main/DatabaseGerakParabola.txt)
3. Program Prediksi : [PrediksiPosisiGerakParabola2.py](https://github.com/FillahAlamsyah/Machine-Learning-Prediksi-Posisi-Pada-Gerak-Parabola/blob/main/PrediksiPosisiGerakParabola2.py)
> [!IMPORTANT]
> Update Program Untuk Prediksi Python. Gunakan file yang dicantumkan di atas.
## Program Membuat Database
```python
import numpy as np
import matplotlib.pyplot as plt

def PosisiParabolaKetikaT(t):
    h0 = 10
    alpha = np.radians(45)
    g = 9.8
    v0 = 10
    v0x = v0*np.cos(alpha)
    v0y = v0*np.sin(alpha)

    ##"Jarak Horizontal Maksimum = ",X," m"
    X = ((v0**2)*np.sin(2*alpha))/(2*g)
    ##"Jarak Vertikal Maksimum = ",Y," m")
    Y = ((v0**2)*(np.sin(alpha)**2))/(2*g)
    ##"Waktu Mencapai Jarak Horizontal Maksimum = ",T," s")
    T = (2*v0*np.sin(alpha))/g     

    #t = np.arange(0.0, T, 0.01)
    y = h0 + v0y*t - 0.5*g*t**2
    x = v0x*t

    print(t,',',round(x,2),',',round(y,2))

print( 't , x , y')
for i in range(0,25):
    t = 0.1*i
    PosisiParabolaKetikaT(round(t,2))
```
## Program Prediksi
```python
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
```
## Output Koordinat
![Gambar Output Koordinat](https://github.com/FillahAlamsyah/Machine-Learning-Prediksi-Posisi-Pada-Gerak-Parabola/blob/main/Output.png?raw=true)

## Hasil Plot Prediksi Parabola
![Gambar Plot Prediksi Terhadap Sesungguhnya](https://github.com/FillahAlamsyah/Machine-Learning-Prediksi-Posisi-Pada-Gerak-Parabola/blob/main/Parabola_Prediction.png?raw=true)
