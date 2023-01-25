# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:32:56 2023

@author: preth
"""

#no of fiction and non fiction               -----
#books published after a particular year ---
#books having a higher rating  ---
#books having a high review  ---
# no of books published by each author --- (sort it ascending)
# cheap books having high ratings ---
#higher rating book are they fiction or non fiction ---
# author with most high rating books ---
#highest rated fiction ---
#highest rated non fiction  ---
#plot per rating
#price of books 
#lowest rated book of a particular author
#display books wrt their rating  ---
# display no of fiction and non fiction       -----


import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("bestsellers with categories.csv")
"""
#1
fiction_count=df['Genre'].value_counts()['Fiction']
nonfiction_count=df['Genre'].value_counts()['Non Fiction']
plotting=pd.DataFrame({'lable':['fiction','non-fiction'],'value':[fiction_count,nonfiction_count]})
plotting.plot.bar(x='lable',y='value',title='No of fiction/non fiction',width=0.1,grid=True,xlabel="Genre",ylabel="No of books")
value=[fiction_count,nonfiction_count]
lable=['fiction','non-fiction']
print(fiction_count,nonfiction_count)
#plt.bar(lable,value)"""
"""

#grouping
def check(grp):
    return grp.mean()
print(df.groupby('Genre')['User Rating'].apply(check))
#df.groupby('Genre')['User Rating'].mean()
print(df.groupby(['Genre','Year'])['User Rating'].mean())
print(df.groupby(['Genre','Year'])['User Rating'].agg(['mean','median']))
print(df.groupby(['Genre','Year'],as_index=False)['User Rating'].mean())


#2
print(df[df['Year']>=2018])

#3
print(df[df['User Rating']>=4.8])
no=df['User Rating'].value_counts()
print(df.groupby('User Rating').size())
print(no)
#plt.bar(df['User Rating'],df['User Rating'].value_counts())

#4
ratings=df.sort_values(by=['User Rating','Reviews'])
print(ratings.tail(1))

#5
print(df.sort_values(by=['Author']).groupby('Author').size())

#6
print(df.sort_values(by=['Price','User Rating'],ascending=['True','False']).head(1))

#7
print(df.sort_values(by=['User Rating']).groupby('Genre').last())

#8
print(df.sort_values(by=['Author']).groupby('Author').size())

#ratting vs no.of books
rating=list(df['User Rating'])
rating.sort()
#print(rating)
uniq=[]
count=[]
for i in rating:
    if str(i) not in uniq:
        uniq.append(str(i))
        count.append(rating.count(i))
#print(uniq)
#print(count)
plt.bar(uniq,count)
plt.bar(df['User Rating'].value_counts().index,df['User Rating'].value_counts())
plt.bar(df['Genre'].value_counts().index,df['Genre'].value_counts())
plt.plot(df['Year'].value_counts().index,df['Year'].value_counts())
"""
plt.scatter(df['User Rating'],df['Price'], color='blueviolet')
plt.title("Ratings vs Price")
plt.xlabel('User Ratings')
plt.ylabel('Price')
plt.show()
plt.plot(df['Year'].value_counts(),'o',mfc='r',mec='b')
"""
pd.set_option('display.max_columns', None)
print(df.head())
print(df[df['User Rating']>=4.8])
print(df.isna().sum())
print(df.describe())
print(df.dtypes)
df.replace('Fiction',1,inplace=True)
df.loc[df['Genre']=='Non Fiction','Genre']=2
print(df.head(10))"""
