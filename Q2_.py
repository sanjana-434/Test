#Q2
#GAUSS SEIDEL
"""
import numpy as np
from sympy import symbols,Eq


A = np.zeros((3,3))
#print("Enter the values of 3x3 matrix : ")

#for i in range(0,3):
#    for j in range(0,3):
#        A[i][j] = float(input())

x,y,z = symbols('x y z')

def diagonallyDominant(eqns):
  #check whether diagonally dominant 
  #else make it diagonally dominant
  if (abs(eqns[0].coeff(x)) > (abs(eqns[0].coeff(y)) + abs(eqns[0].coeff(z))) 
  and abs(eqns[1].coeff(x)) > (abs(eqns[1].coeff(y)) + abs(eqns[1].coeff(z)))  
  and abs(eqns[2].coeff(x)) > (abs(eqns[2].coeff(y)) + abs(eqns[2].coeff(z))) ):
    return eqns
  else:
    e = [0,0,0]  #list to reaarange to make it dd
    for i in range(0,3):
      if ((eqns[i].coeff(x))**2 > ((eqns[i].coeff(y))**2 + (eqns[i].coeff(z))**2)):
        e[0] = (eqns[i])
        break
    for i in range(0,3):
      if ((eqns[i].coeff(y))**2 > ((eqns[i].coeff(x))**2 + (eqns[i].coeff(z))**2)):
        e[1] = (eqns[i])
        break
    for i in range(0,3):
      if ((eqns[i].coeff(z))**2 > ((eqns[i].coeff(x))**2 + (eqns[i].coeff(y))**2)):
        e[2] = (eqns[i])
        break 
    return e 

def seidel(x_,y_,z_,eqns):
    x_ = (-(eqns[0] - eqns[0].coeff(x)*x - eqns[0].coeff(y)*y - eqns[0].coeff(z)*z) - eqns[0].coeff(z)*z_ - eqns[0].coeff(y)*y_ )/eqns[0].coeff(x)
    y_ = (-(eqns[1] - eqns[1].coeff(x)*x - eqns[1].coeff(y)*y - eqns[1].coeff(z)*z) - eqns[1].coeff(x)*x_ - eqns[1].coeff(z)*z_ )/eqns[1].coeff(y)
    z_ = (-(eqns[2] - eqns[2].coeff(x)*x - eqns[2].coeff(y)*y - eqns[2].coeff(z)*z) - eqns[2].coeff(x)*x_ - eqns[2].coeff(y)*y_ )/eqns[2].coeff(z)

    return x_,y_,z_

e1 = 26*x+2*y+2*z-12.6
e2 = 3*x+27*y+z+14.3
e3 = 2*x+3*y+17*z-6.0

eqns = [e1,e2,e3]
eqns = diagonallyDominant(eqns)
x_ = 0
y_ = 0
z_ = 0
for i in range(0,10):
    x_,y_,z_ = seidel(x_,y_,z_,eqns)
    #print(x_,y_,z_)
print("Values of x,y and z : ")
print(x_.round(2),y_.round(2),z_.round(2))
"""
#GAUSS JACOBI
import numpy as np
from sympy import symbols,Eq

A = np.zeros((3,3))
#print("Enter the values of 3x3 matrix : ")
"""
for i in range(0,3):
    for j in range(0,3):
        A[i][j] = float(input())
"""
x,y,z = symbols('x y z')
def diagonallyDominant(eqns):
  #check whether diagonally dominant 
  #else make it diagonally dominant
  if (abs(eqns[0].coeff(x)) > (abs(eqns[0].coeff(y)) + abs(eqns[0].coeff(z))) 
  and abs(eqns[1].coeff(x)) > (abs(eqns[1].coeff(y)) + abs(eqns[1].coeff(z)))  
  and abs(eqns[2].coeff(x)) > (abs(eqns[2].coeff(y)) + abs(eqns[2].coeff(z))) ):
    return eqns
  else:
    e = [0,0,0]  #list to reaarange to make it dd
    for i in range(0,3):
      if ((eqns[i].coeff(x))**2 > ((eqns[i].coeff(y))**2 + (eqns[i].coeff(z))**2)):
        e[0] = (eqns[i])
        break
    for i in range(0,3):
      if ((eqns[i].coeff(y))**2 > ((eqns[i].coeff(x))**2 + (eqns[i].coeff(z))**2)):
        e[1] = (eqns[i])
        break
    for i in range(0,3):
      if ((eqns[i].coeff(z))**2 > ((eqns[i].coeff(x))**2 + (eqns[i].coeff(y))**2)):
        e[2] = (eqns[i])
        break 
    return e

def jacobi(x_,y_,z_,eqns):
    a = (-(eqns[0] - eqns[0].coeff(x)*x - eqns[0].coeff(y)*y - eqns[0].coeff(z)*z) - eqns[0].coeff(z)*z_ - eqns[0].coeff(y)*y_ )/eqns[0].coeff(x)
    b = (-(eqns[1] - eqns[1].coeff(x)*x - eqns[1].coeff(y)*y - eqns[1].coeff(z)*z) - eqns[1].coeff(x)*x_ - eqns[1].coeff(z)*z_ )/eqns[1].coeff(y)
    c = (-(eqns[2] - eqns[2].coeff(x)*x - eqns[2].coeff(y)*y - eqns[2].coeff(z)*z) - eqns[2].coeff(x)*x_ - eqns[2].coeff(y)*y_ )/eqns[2].coeff(z)
    return a,b,c

e1 = 26*x+2*y+2*z-12.6
e2 = 3*x+27*y+z+14.3
e3 = 2*x+3*y+17*z-6.0

eqns = [e1,e2,e3]
eqns = diagonallyDominant(eqns)
x_ = 0
y_ = 0
z_ = 0
for i in range(0,10):
    x_,y_,z_ = jacobi(x_,y_,z_,eqns)
    #print(x_,y_,z_)
print("Values of x,y and z : ")
print(x_.round(2),y_.round(2),z_.round(2))

