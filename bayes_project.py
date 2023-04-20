# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:55:03 2023

@author: Gabriela Hilario Acuapan
File:
Comments:
----------------------------------------
"""


#Cargar librerías
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm

# Cargar la base de datos
database = np.loadtxt('Database.csv', delimiter=',')

# Diccionarios
Dir_classes = {'setosa':0, 'versicolor':1}

# Separar en clases
class1 = database[0:40,:]
class2 = database[50:90,:]

# Features
c1_x1 = class1[:,0]
c1_x2 = class1[:,1]
c1_x3 = class1[:,2]
c1_x4 = class1[:,3]

c2_x1 = class2[:,0]
c2_x2 = class2[:,1]
c2_x3 = class2[:,2]
c2_x4 = class2[:,3]

# --------------------- CLASS 1 -----------------------
# Mean
c1_mean_x1 = statistics.mean(c1_x1)
c1_mean_x2 = statistics.mean(c1_x2)
c1_mean_x3 = statistics.mean(c1_x3)
c1_mean_x4 = statistics.mean(c1_x4)

# --------------------- CLASS 2 -----------------------
# Mean
c2_mean_x1 = statistics.mean(c2_x1)
c2_mean_x2 = statistics.mean(c2_x2)
c2_mean_x3 = statistics.mean(c2_x3)
c2_mean_x4 = statistics.mean(c2_x4)

# -------------------- Varianzas --------------------

c1_x1_sorted = sorted(c1_x1)
c1_x2_sorted = sorted(c1_x2)
c1_x3_sorted = sorted(c1_x3)
c1_x4_sorted = sorted(c1_x4)

c2_x1_sorted = sorted(c2_x1)
c2_x2_sorted = sorted(c2_x2)
c2_x3_sorted = sorted(c2_x3)
c2_x4_sorted = sorted(c2_x4)


c1_var_x1 = statistics.variance(c1_x1_sorted)
c1_var_x2 = statistics.variance(c1_x2_sorted)
c1_var_x3 = statistics.variance(c1_x3_sorted)
c1_var_x4 = statistics.variance(c1_x4_sorted)

c2_var_x1 = statistics.variance(c2_x1_sorted)
c2_var_x2 = statistics.variance(c2_x2_sorted)
c2_var_x3 = statistics.variance(c2_x3_sorted)
c2_var_x4 = statistics.variance(c2_x4_sorted)

# desviaciones

c1_des_x1 = np.std(c1_x1_sorted)
c1_des_x2 = np.std(c1_x2_sorted)
c1_des_x3 = np.std(c1_x3_sorted)
c1_des_x4 = np.std(c1_x4_sorted)

c2_des_x1 = np.std(c2_x1_sorted)
c2_des_x2 = np.std(c2_x2_sorted)
c2_des_x3 = np.std(c2_x3_sorted)
c2_des_x4 = np.std(c2_x4_sorted)

# Distribuciones P(w|wi)
Px1_w1 = norm.pdf(c1_x1_sorted, c1_mean_x1, c1_var_x1)
Px2_w1 = norm.pdf(c1_x2_sorted, c1_mean_x2, c1_var_x2)
Px3_w1 = norm.pdf(c1_x3_sorted, c1_mean_x3, c1_var_x3)
Px4_w1 = norm.pdf(c1_x4_sorted, c1_mean_x4, c1_var_x4)


# Distribuciones P(w|wi)
Px1_w2 = norm.pdf(c2_x1_sorted, c2_mean_x1, c2_var_x1)
Px2_w2 = norm.pdf(c2_x2_sorted, c2_mean_x2, c2_var_x2)
Px3_w2 = norm.pdf(c2_x3_sorted, c2_mean_x3, c2_var_x3)
Px4_w2 = norm.pdf(c2_x4_sorted, c2_mean_x4, c2_var_x4)

### ---------- gáficas
# plt.figure(1)
# plt.plot(c1_x1_sorted, norm.pdf(c1_x1_sorted, c1_mean_x1, c1_var_x1))
# plt.grid(True)
# plt.figure(2)
# plt.plot(c1_x2_sorted, norm.pdf(c1_x2_sorted, c1_mean_x2, c1_var_x2))
# plt.grid(True)
# plt.figure(3)
# plt.plot(c1_x3_sorted, norm.pdf(c1_x3_sorted, c1_mean_x3, c1_var_x3))
# plt.grid(True)
# plt.figure(4)
# plt.plot(c1_x4_sorted, norm.pdf(c1_x4_sorted, c1_mean_x4, c1_var_x4))
# plt.grid(True)
# plt.show()
#---------------
# plt.figure(1)
# plt.plot(c2_x1_sorted, norm.pdf(c2_x1_sorted, c2_mean_x1, c2_var_x1))
# plt.grid(True)
# plt.figure(2)
# plt.plot(c2_x2_sorted, norm.pdf(c2_x2_sorted, c2_mean_x2, c2_var_x2))
# plt.grid(True)
# plt.figure(3)
# plt.plot(c2_x3_sorted, norm.pdf(c2_x3_sorted, c2_mean_x3, c2_var_x3))
# plt.grid(True)
# plt.figure(4)
# plt.plot(c2_x4_sorted, norm.pdf(c2_x4_sorted, c2_mean_x4, c2_var_x4))
# plt.grid(True)
# plt.show()


### 
plt.figure(1)
plt.plot(c1_x1_sorted, norm.pdf(c1_x1_sorted, c1_mean_x1, c1_var_x1), label='$w_1$')
plt.plot(c2_x1_sorted, norm.pdf(c2_x1_sorted, c2_mean_x1, c2_var_x1), label='$w_2$')
plt.ylabel('P(x|$w_i$)')
plt.xlabel('x')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(c1_x2_sorted, norm.pdf(c1_x2_sorted, c1_mean_x2, c1_var_x2), label='$w_1$')
plt.plot(c2_x2_sorted, norm.pdf(c2_x2_sorted, c2_mean_x2, c2_var_x2), label='$w_2$')
plt.ylabel('P(x|$w_i$)')
plt.xlabel('x')
plt.legend()
plt.grid(True)

plt.figure(3)
plt.plot(c1_x3_sorted, norm.pdf(c1_x3_sorted, c1_mean_x3, c1_var_x3), label='$w_1$')
plt.plot(c2_x3_sorted, norm.pdf(c2_x3_sorted, c2_mean_x3, c2_var_x3), label='$w_2$')
plt.ylabel('P(x|$w_i$)')
plt.xlabel('x')
plt.legend()
plt.grid(True)

plt.figure(4)
plt.plot(c1_x4_sorted, norm.pdf(c1_x4_sorted, c1_mean_x4, c1_var_x4), label='$w_1$')
plt.plot(c2_x4_sorted, norm.pdf(c2_x4_sorted, c2_mean_x4, c2_var_x4), label='$w_2$')
plt.ylabel('P(x|$w_i$)')
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.show()


## Matriz de covarianza

Cov_class1 = np.cov(class1.astype(float), rowvar=False)
Cov_class1 = np.cov(class1.astype(float), rowvar=False)
#print(Cov_class1)