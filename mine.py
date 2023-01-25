import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("test.csv")


import piplite
await piplite.install('seaborn')
import seaborn as sns

print(df.head())

print(df.info())

print(df["LotFrontage"].describe().round(2))

df.set_index(df["Id"])

print(df["KitchenQual"].unique())
df["KitchenQual"] = df["KitchenQual"].replace("TA","Ex")
print(df["KitchenQual"].head())
print(df["KitchenQual"].unique())

print(df.duplicated().sum())
df.drop_duplicates()

s = df.loc[df["Alley"].notnull(),"Alley"]

print(s)

df["Alley"] = df["Alley"].fillna(method = 'bfill')
df["Alley"] = df["Alley"].fillna(method = 'ffill')
df["Alley"]
df.dropna(how = "all")

print(df["Fence"].isnull().sum())

print(df["LotFrontage"].describe()['25%'])
df["LotFrontage"].plot(kind = 'box',vert = False)

fig = plt.figure(figsize = (10,6))
plt.boxplot(x = df["LotFrontage"],vert = False)
plt.show()

sns.distplot(df["LotFrontage"],kde = False)

sns.distplot(df["LotFrontage"],hist =False)

df["KitchenQual"].value_counts().plot(kind ='pie',colors = ["pink","green","yellow","orange"],figsize = (2,2))

ax = df["KitchenQual"].value_counts().plot(kind ='bar',figsize = (2,2))
ax.set_xlabel("KitchenQual")
ax.set_title("kjern")

#stacked bar plot
x1 = df["KitchenQual"].value_counts().index
y1=df["KitchenQual"].value_counts()
#x2 = df["Electrical"]
y2= np.array([34,56,76])

plt.bar(x1,y1)
plt.bar(x1 ,y2,bottom = y1)
plt.show()

#x1 = df["KitchenQual"].value_counts().index
x1 = np.array([0,1,2])
y1=df["KitchenQual"].value_counts()
#x2 = df["Electrical"]
y2= np.array([34,56,76])


plt.bar(x1-0.2,y1,width = 0.4)
plt.bar(x1+0.2 ,y2,width = 0.4)
plt.xticks(ticks = x1,labels = np.array(df["KitchenQual"].value_counts().index))
plt.show()


fig,ax = plt.subplots()
sns.distplot(df["LotFrontage"])
ax.axvline(df["LotFrontage"].mean(),color = 'r')
plt.show()

#s = (df.loc[df["LotFrontage"].notnull(),"LotFrontage"])
#print(s)

#s = df["LotFrontage"].dropna()
#s

i = df[df["LotFrontage"].isnull()].index
df.drop(i,inplace = True)


print(df.head())
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())
print(df.head())
print(df["LotFrontage"].isnull().sum())

corr = df.corr()
col = df.select_dtypes(include = ['int64']).columns
print(col)
for i in col:
    for j in col:
        if (i!=j):
            if (corr[i][j] >0.8):
                print(i ," [ ] " ,j ,corr[i][j])

h = sns.heatmap(corr,cmap="YlGnBu")
h.set_title("Correlation",fontdict = {"fontsize":10})

x_ = df["2ndFlrSF"]
x_ = np.array(x_)
print(x_)
print(x.isnull().sum())

y_ = df["GrLivArea"]
y_ = np.array(y_)
print(y.isnull().sum())
print(y_)

sample = df[["2ndFlrSF","GrLivArea"]]
print(sample.corr())

df.plot(kind = 'scatter',x = "2ndFlrSF",y="GrLivArea")
#plt.scatter(x_,y_)
plt.show()

reg = np.polyfit(x_,y_,deg = 1)
reg

trend = np.polyval(reg,x_)
print(trend)

plt.plot(x_,trend,color = 'r',label = "y_pred")
plt.scatter(x_,y_,label = "y")
plt.legend()

plt.show()

np.polyval(reg,[200,800])


plot_color_gradients('Perceptually Uniform Sequential',
                     ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

plot_color_gradients('Sequential',
                     ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

plot_color_gradients('Diverging',
                     ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])

plot_color_gradients('Qualitative',
                     ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c'])