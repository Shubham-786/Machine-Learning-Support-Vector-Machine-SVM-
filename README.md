# Machine-Learning-Support-Vector-Machine-SVM-

#WHAT IS SUPPORT VECTOR MACHINE ALGORITHMS?

It is widely used classification algorithms.
The idea of SVM is simple:- This algorithm creates a separation line which divides the classes in the best possible manner.

For Example:- Dog or Cat, diseases or no diseases.

PROBLEM STATEMENT:- Classify muffin and cupcake receipe using SVM.

DATASET:

Type	Flour	Milk	Sugar	Butter	Egg	Baking Powder	Vanilla	Salt
Muffin	55	 28	   3	    7	     5	       2	       0	    0
Muffin	47	 24  	 12	    6	     9	       1	       0	    0
Muffin	47	 23	   18	    6	     4	       1	       0	    0
Muffin	45	 11	   17	    17   	 8	       1	       0	    0
Muffin	50	 15 	 12    	6	     5	       2	       0	    0
Muffin	49	 10	   6	    7	     4	       1	       1	    0
Cupcake	39	 0	   19	   10	     14	       1	       1	    0
Cupcake	42	 22	   20	   19	     8	       3	       1	    0
Cupcake	34	 21	   26	   10	     5	       2	       1	    0
Cupcake	31	 17	   16	   20	     7	       1	       0	    0
Cupcake	30	 15	   22	   22	     3	       2	       0	    0
Cupcake	29	 11	   24	   17	     2	       4	       1	    0
Cupcake	39	 17	   27	   19	     1	       2	       1	    0

WHAT IS DIFFERENCE BETWEEN MUFFIN and CUPCAKE?

Turns out muffins have more flour, while cupcakes have more butter and sugar.

#Packages for Data Analysis
import numpy as np
import pandas as pd

from sklearn import svm

#Visualize our Data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

%matplotlib inline
#it will draw our graphs in same page in Jupyter notebook.

receipes = pd.read_csv('muffin-or-cupcake.csv')
print(receipes.head())
#print 1st 5 line of data

#Plot the Data
sns.lmplot('Flour','Sugar',data=receipe, hue='Type', palette='set1',fit_reg = false, scatter_kws={"s":70})

#Format or Pre-Processing our Data
Type_label = np.where(receipe['Type']=='Muffin',0,1)
receipe_features = receipes.coloumns.values[1:].tolist()
receipe_features

#Output: ['Flour','Milk','Sugar','Butter','Egg','Baking Powder','Vanilla','Salt']

ingredients = receipes[['Flour','Sugar']].values
print(ingredients)
#It will print output values for Flour and Sugar as we are only creating for Flour and Sugar if we want to print for all then we need to add as ingredients = receipes[[receipe_features]].values

#Fitting The Model
Model = svm.SVC(Kernal='Linear')
Model.fit(ingredients, type_label)

#Get the Seperating Hyperlane
w = Model.coef_[0]
a = -w[0] / w[1]
#Since we are generating slope
xx = np.linespace(30,60)
yy = a*xx-(Model.intercept-[0])/w[1]

#Plot the parallels to the seperating hyperlane that pass through the support vectors.
b = model.support_vectors_[0]
yy_down = a*xx + (b[1]-a*b[0])
b = model.support_vectors_[-1]
yy_up = a*xx +(b[1]-a*b[0])

#Plot
sns.lmplot('Flour','Sugar',Data=receipe,hue='Type',palette='set1',fit_reg=false,scatter_kws={"s":70})
plt.plot(xx,yy,linewidth=2,color='black')
plt.plot(xx,yy_down,'k--')
#k-- is for dotted line on graph
plt.plot(xx,yy_up,'k--')

#Create a Function to predict muffin or cupcake
def muffin_or_cupcake(Flour,Sugar):
if(model.predict([[Flour,Sugar]]))==0:
print('You\re looking at Muffin receipe!')
else:
print('You\re looking at Cupcake receipe!')

#Predict if 50 parts Flour and 20 part Sugar
muffin_or_cupcake(50,20)

#Output:- You are looking at Muffin Receipe!

#Lets Plot this on Graph
sns.lmplot('Flour','Sugar',Data=receipe,hue='type',palette='set1',fit_reg=false,scatter_kws={"s":70})
plt.plot(xx,yy,linespace=2,color='black')
plt.plot(50,20,'yo',markersize='9')
#yo is yellow color to denote in graph

#HENCE, WE HAVE BUILT A CLASSIFIER USING SVM WHICH IS ABLE TO CLASSIFY IF A RECEIPE IS OF A CUPCAKE OR A MUFFIN!
