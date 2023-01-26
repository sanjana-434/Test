"""
import numpy as np
from sympy import symbols,Eq,Derivative

#Q1
#BISECTION METHOD
x = symbols('x')
y = x**3 -4*x -9

def f(a): 
    c = y.subs(x,a)
    return c

def find_endpoints():
  a = 0 
  b = 1
  while(1):
    y1 = f(a)
    y2 = f(b)
    #print(y1,y2)
    if ((y1<0 and y2>0) or (y1>0 and y2<0)):
      break
    elif (y1<0 and y2<0):
      a +=1
      b+=1
    else:
      a-=1
      b-=1
  return a,b

a,b = find_endpoints()
#print(a,b)

for i in range(0,20):
    avg = (a+b)/2
    f_avg = f(avg)
    if (f_avg < 0):
        a = avg
    elif (f_avg > 0):
        b =  avg
print("%.3f"%avg)


#REGULA-FALSI METHOD
import numpy as np
from sympy import *


x = symbols('x')
y = x**3-x-1

def f(a):
    c = y.subs(x,a)
    return c

def find_endpoints():
  a = 0
  b = 1
  while(1):
    y1 = f(a)
    y2 = f(b)
    #print(y1,y2)
    if (y1<0 and y2>0):
      break
    elif (y1<0 and y2<0):
      a +=1
      b+=1
    else:
      a-=1
      b-=1
  return a,b

a,b = find_endpoints()

for i in range(0,10):
    f_a = f(a)
    f_b = f(b)
    #print(f_a,f_b)
    t = (f_b * a - f_a * b)/(f_b-f_a)
    if (t < 0):
        a = t
    elif (t > 0):
        b =  t
print(t.round(3))
"""

#NEWTON-RAPHSON
import numpy as np
from sympy import *

x = symbols('x')
y = x**2-2
y_diff = Derivative(y,x).doit()


def f(a):
    c = y.subs(x,a)
    return c
def f_diff(a):
    c = y_diff.subs(x,a)
    return c

def find_endpoints():
  a = 0
  b = 1
  while(1):
    y1 = f(a)
    y2 = f(b)
    #print(y1,y2)
    if ((y1<0 and y2>0) or (y1>0 and y2<0)):
      break
    elif (y1<0 and y2<0):
      a +=1
      b+=1
    else:
      a-=1
      b-=1
  return a,b

a,b = find_endpoints()
#print(a,b)
#print(f(a),f(b))

x_ = (a+b)/2
#print(y_diff)
for i in range(0,20):
    x_ = x_ - f(x_)/f_diff(x_)
    #print(x_)
print("%.3f"%x_)

