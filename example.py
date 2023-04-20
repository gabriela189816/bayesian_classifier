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
class1 = database[0:50,:]
class2 = database[50:100,:]


# Features
sepal_length = database[:,0]
sepal_width = database[:,1]
petal_length = database[:,2]
petal_width = database[:,3]

c1_x1 = sepal_length[0:49]
c1_x2 = sepal_width[0:49]
c1_x3 = petal_length[0:49]
c1_x4 = petal_width[0:49]

c2_x1 = sepal_length[50:99]
c2_x2 = sepal_width[50:99]
c2_x3 = petal_length[50:99]
c2_x4 = petal_width[50:99]

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


# c1_var_x1 =  (sepal_length[0:49]-c1_mean_x1)**2
# c1_var_x2 =  (sepal_width[0:49]-c1_mean_x2)**2
# c1_var_x3 =  (petal_length[0:49]-c1_mean_x3)**2
# c1_var_x4 =  (petal_width[0:49]-c1_mean_x4)**2

# c2_var_x1 =  (sepal_length[50:99]-c2_mean_x1)**2
# c2_var_x2 =  (sepal_width[50:99]-c2_mean_x2)**2
# c2_var_x3 =  (petal_length[50:99]-c2_mean_x3)**2
# c2_var_x4 =  (petal_width[50:99]-c2_mean_x2)**2



# x = np.arange(0,10,0.1)
# y = x*np.cos(x)

# plt.plot(sepal_length[0:49],y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Lab DLS')
# plt.show()

## Matriz de covarianza

c1_x1_sorted = sorted(c1_x1)
c1_x2_sorted = sorted(c1_x2)
c1_x3_sorted = sorted(c1_x2)
c1_x4_sorted = sorted(c1_x2)

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

plt.figure(1)
plt.plot(c1_x1_sorted, norm.pdf(c1_x1_sorted, c1_mean_x1, c1_var_x1))
plt.figure(2)
plt.plot(c1_x2_sorted, norm.pdf(c1_x2_sorted, c1_mean_x2, c1_var_x2))
plt.figure(3)
plt.plot(c1_x3_sorted, norm.pdf(c1_x3_sorted, c1_mean_x3, c1_var_x3))
plt.figure(4)
plt.plot(c1_x4_sorted, norm.pdf(c1_x4_sorted, c1_mean_x4, c1_var_x4))
plt.show()



Cov_class1 = np.cov(class1.astype(float), rowvar=False)
Cov_class1 = np.cov(class1.astype(float), rowvar=False)
#print(Cov_class1)