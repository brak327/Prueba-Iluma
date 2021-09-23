#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# leer el archivo en el pc
car ='C:/Users/brak3/OneDrive/Escritorio/Iluma/carcasses.xlsx'
fin ='C:/Users/brak3/OneDrive/Escritorio/Iluma/finishers.xlsx'
nec ='C:/Users/brak3/OneDrive/Escritorio/Iluma/necropsies.xlsx'


# In[2]:


#Definir los DataFrame que vamos a utilizar
dfcar = pd.read_excel(car)
dffin = pd.read_excel(fin)
dfnec = pd.read_excel(nec)


# In[3]:


# chequeamos cuantos son los atributos y cuantos registros de cada DF
print (dfcar.shape)
print (dffin.shape)
print (dfnec.shape)


# In[4]:


# Se hace chequeo de cuantos registros en los 3 DF tiene entradas nulas. 
# Como no tenemos valores 'NaN' o '?', proseguimos con el analisis
missing_datacar = dfcar.isnull()
for column in missing_datacar.columns.values.tolist():
    print(column)
    print (missing_datacar[column].value_counts())
    print("")
    
missing_datafin = dffin.isnull()
for column in missing_datafin.columns.values.tolist():
    print(column)
    print (missing_datafin[column].value_counts())
    print("")
    
missing_datanec = dfnec.isnull()
for column in missing_datanec.columns.values.tolist():
    print(column)
    print (missing_datanec[column].value_counts())
    print("")


# Comparamos la cantidad de informacion obetenida luego de unir los DF de acuerdo a ciertas filas que comparten. 
# Podremos ver que la disminucion es considerable. 

# In[5]:


# 1. Unir Finishers con carcasses con puntos comun fecha y granja
dfmerge1 = dffin.merge(dfcar,left_on=["fechaSalida", "granjaDeOrigen"],right_on = ["fecha", "granja"])
dfmerge1.shape
print ("el numero de registros que concuerdan es de:", dfmerge1.shape ,
       "Se nota una reduccion de las entradas de hasta el 80%")


# <b>Se basara la union de los DF en las siguientes afirmaciones dadas en la descripción del problema.<b>
#     
#     . Es válido asumir que los cerdos llegan a la planta de procesamiento el mismo día que salen de la granja.
#     . La fecha que se reporta en los registros de operaciones es la fecha promedio de salida de los animales (finishers). 
#     . En cambio, los registros de procesamiento sí reflejan la fecha exacta en la que los cerdos llegaron a la planta (Cascasses).
# 

# In[6]:


# Unir Finishers con Carcasses con la suposicion echa al inicio, fecha de salia en Finishers con la fecha de de Carcasses
dfmerge2 = dffin.merge(dfcar,left_on="fechaSalida",right_on = "fecha")
dfmerge2.shape
print ("el numero de registros que coinciden es de:" , dfmerge2.shape)
print (len(dfmerge2.index), "equivale al 67.9% de los registros, haciendolo la mejor base para realizar el analisis")


# In[7]:


# 1 (Afirmación David)
#Se unen en un solo DF las variables que David creo que influiran y buscamos la posible correlación
dfdavid = dfmerge2 [["Edad", "magroPor", "numeroCanales"]]
print (dfdavid.corr())
print ("Los valores de correlacion son bastante bajos, lo que no indica una relación directa o por lo menos linearentre las variables")


# In[8]:


sns.regplot(x="grasaDorsalMm", y="magroPor", data=dfmerge2)


# In[9]:


# 2 (Afirmación Alejandro)
# Creamos un DF alterno que nos permita analizar lo que dice Alejandro
dfalejandro = dfmerge2 [["gananciaAnimalDiaKg", "consumoAnimalDiaKg"]]
print (dfalejandro.corr())
sns.regplot(x="consumoAnimalDiaKg", y="gananciaAnimalDiaKg", data=dfalejandro)


# In[10]:


# 3 (Afirmacion Alejandro)
# Se evalua la relacion con correlacion de Pearson, los coeficientes de Pearson nos lo aclaran de inmediato

from scipy import stats

pearson_coef, p_value = stats.pearsonr(dfalejandro['consumoAnimalDiaKg'], dfalejandro['gananciaAnimalDiaKg'])
print("Es coeficiente de correlacion de Pearson es", pearson_coef, " con un valor de P igua a P =", p_value) 

print ("al ser un valor de P menor de 0.001 la relacion es estaditicamente significante apesar que la relacion lineal no sea muy fuerte")


# In[11]:


# 3 evaluar la relacion lineal encontrada entre consumo animal y ganancia, para encontrar el promedio de grasa dorsal en Mm

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
lm.fit(dfalejandro, dfmerge2['grasaDorsalMm'])
print (lm.intercept_)
print (lm.coef_)


# Los coeficientes de Pearson son sugieren una fuerte correlacion, podria aumentar con la calidad de los datos. La relacion hallada al ser lineal no poseera variables de mayor orden a uno. Dependera de "consumo al dia" y "ganancia" arrojando lo siguiente:
# 
# <b>Grasa dorsal (Mm) = 7.45 + 15.40(Consumo al dia) - 4.12(Ganancia al dia)]<B>

# In[12]:


# 4 (Afirmacion Miguel)
# Miguel Afirma que debemos centrarnos en producir la mayor cantidad de Kg de carne magra por cerdo
# Unimos en un nuevo DF las variables que tienen correlacion con carne magra
dfmiguel = dfmerge2 [["%Ganancia", "Edad" , "pesoCanal", "grasaDorsalMm", "consumoAnimalDiaKg" , "magroPor", 
                      "gananciaAnimalDiaKg"]]


# In[13]:


# 4.4 miguel 
#Para tener una mejor prediccion de porcentage magro, nos limitamos a evaluarla con las variables dependientes, %ganancia
# Edad, Peso y grasa dorsal. Puede que no nos lleven a una expresion con error BAJO, pero puede acercarnos a los valores
# de Magro durante los dias en los que los cerdos muestran la conversion alimenticia mas alta
dfmiguel2 = dfmiguel[dfmiguel.Edad.isin([100,101,102,103,104,105,106,107])]
print ( dfmiguel2.head())
print ("EL dataFrame Resultante tiene las siguientes dimensiones" , dfmiguel2.shape)


# In[14]:


# 4 (Miguel)
# Creamos un Df con las variables correlacionas con Magro
miguelev = dfmiguel2[["pesoCanal","Edad","%Ganancia","grasaDorsalMm","consumoAnimalDiaKg","gananciaAnimalDiaKg"]]
# Con magroev se evaluara la funcion predictiva
magroev = dfmiguel2["magroPor"]


# In[15]:


#  4 (Miguel) analisis lineal para magro con respecto a las otras variables 
lm2 = LinearRegression()
lm2
lm2.fit(miguelev,dfmiguel2["magroPor"])
print (lm2.intercept_)
print (lm2.coef_)


# In[16]:


# 4 (miguel) Funcion predictiva
Y_hat = lm2.predict(miguelev)


# In[17]:


# 4 (Miguel)
width = 12
height = 10
plt.figure(figsize=(width, height))


ax1 = sns.distplot(dfmiguel2['magroPor'], hist=False, color="r", label="Valor Actual")
sns.distplot(Y_hat, hist=False, color="b", label="Valores ajustados" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('magro proporcion')
plt.ylabel('proporcion cerdos')

plt.show()
plt.close()


# Podemos ver por la grafica que la mejor prediccion la proporcion magra del canal "magroPor" que podemos obtener (curva azul) es con una relacion de las variables que nos mostraron correlacion con esta, resultando asi en la siguiente ecuacion:
# 
# <b>magroPor = 0.6197 + 4.5607e-05(pesoCanal) - 9.7098e-06(Edad) + 4.4438e-04(%Ganancia) - 7.3102e-03(grasaDorsalMm) + 7.9262e-03(consumoAnimalDiaKg) - 2.0427e-02(gananciaAnimalDiaKg)<b>
#     
# Asi como asegura Guillermo, al tener mas carne magra (Qué es la que mas se vende y a mejor precio) obtendremos mayores ingresos. Los costos fijos y variables se diluyen. Pero no se debe ignorar el hecho que los otros indicadores productivos SI influyen en el resultado final de la canal a vender. 

# #Jaime y miguel
# Vamos a corroborar con ambas tablas (Finishers y Carcasses) si los pesos que reportan ambas personas coinciden.
# Las unimos por medio de "fechaSalida" y "pesoCanalSupuesto" en finishers y en Carcasses con "fecha" y "pesoCanal"
# Asumiendo que el 14% del perso en pie del cerdo se pierde en el sacrificio 
# <b>pesoCanalSupuesto = pesoPromedioFinalKg * 0.86<B>

# In[18]:


#Se verifica la cantadidad de registros que poseemos entre el peso final en pie del cerdo comprado
# con el peso de la canal
dfmergefincar = dffin.merge(dfcar,left_on=["fechaSalida", "pesoCanalSupuesto"],right_on = ["fecha", "pesoCanal"])
dfmergefincar.shape
print ("el numero de registros que concuerdan es de:", dfmergefincar.shape)


# In[19]:


get_ipython().run_cell_magic('capture', '', '! pip install ipywidgets\nfrom IPython.display import display\nfrom IPython.html import widgets \nfrom IPython.display import display\nfrom ipywidgets import interact, interactive, fixed, interact_manual')


# In[20]:


#Solo usamos la parte numerica del DF para no entrar en valores que no pueden ser calculados por la variables categoricas
dfmerge3=dfmerge2._get_numeric_data()
dfmerge3.head()


# In[21]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('magroPor')
    plt.ylabel('proporcion de cerdos')

    plt.show()
    plt.close()


# In[22]:


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #datos de entrenamiento 
    #datos para evaluar
   
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Funcion predictiva')
    plt.ylim([-10000, 60000])
    plt.ylabel('MagroPor')
    plt.legend()


# In[23]:


y_data = dfmerge3['magroPor']
# Eliminamos la columna Magro ya que es el valor que queremos calcular
x_data=dfmerge3.drop('magroPor',axis=1)


# In[24]:


# Seleccionamos al azar porcentajes de los datos para entrenar y probar la ecuacion de prediccion que se averiguo en
# el enunciado de miguel
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)


print("numero de registros para test :", x_test.shape[0])
print("numero de registros para entrenar:",x_train.shape[0])


# In[25]:


lre=LinearRegression()
lre.fit(x_train[['grasaDorsalMm']], y_train)


# In[26]:


lre.score(x_test[['grasaDorsalMm']], y_test)
# El valor R^2 es mas cercano a 1 con lo que podemos
lre.score(x_train[['grasaDorsalMm']], y_train)


# Al poseer un correlacion buena, podremos usarlos como prediccion en nuestro modelo y modificarlo entrenando el mismo modelo

# In[27]:


from sklearn.model_selection import cross_val_score
# se crea un objeto para hacer regresion y predecir magro con mas variables
lr = LinearRegression()
lr.fit(x_train[['grasaDorsalMm', '%Ganancia', 'consumoAnimalDiaKg', 'pesoCanal']], y_train)


# In[28]:


# Se dividen los set de datos para entrenar la funcion de prediccion
# Usando el parte de entrenamiento haremos prediccion del valores de magroPor
yhat_train = lr.predict(x_train[['grasaDorsalMm', '%Ganancia', 'consumoAnimalDiaKg', 'pesoCanal']])
yhat_train[0:5]


# In[29]:


# Usaremos la parte de evaluacion para predecir valores de magroPor
yhat_test = lr.predict(x_test[['grasaDorsalMm', '%Ganancia', 'consumoAnimalDiaKg', 'pesoCanal']])
yhat_test[0:5]


# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[31]:


Title = 'grafica de distribucion entre valores predichos por el set de datos de entrenamiento vs el set de datos reales de entrenamiento'
DistributionPlot(y_train, yhat_train, "valores obtenidos (Entrenamiento)", "valores reales (Entrenamiento)", Title)


# In[32]:


Title = 'grafica de distribucion entre valores predichos por el set de datos de evaluacion vs los valores del set de datos de evaluacion'
DistributionPlot(y_test, yhat_test, "valores obtenidos (Evaluacion)", "valores reales (Evaluacion)", Title)


# Ya que ambos set de datos se comportan acordes a los valores reales, no es necesario buscar nuevos valores para evaluar la funcion de prediccion. Este modelo no ha tomado tanto ruido de datos que pueda que no sean tomados de la mejor manera.
