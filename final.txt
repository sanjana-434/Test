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

#1b
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

#1c
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

#2b
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

#Q3
#POWER METHOD
import numpy as np

#3x3 matrix
A = np.zeros((3,3))
#print("Enter the values of 3x3 matrix : ")

#for i in range(0,3):
#    for j in range(0,3):
#        A[i][j] = float(input())

A[0][0] = 1
A[0][1] = 2
A[0][2] = 0
A[1][0] = -2
A[1][1] = 1
A[1][2] = 2 
A[2][0] = 1
A[2][1] = 3
A[2][2] = 1

X_initial = np.array([[1,1,1]])
#X_initial.shape[0]

def mul(A,B):
    result = np.zeros((3,1))
    #print(A,B,result)
    for i in range(0,A.shape[0]):
        for j in range(0,B.shape[1]):
            for k in range(0,B.shape[0]):
                result[i][j] += A[i][k]*B[k][j]
    #print("Result : ",result)
    return result

X = X_initial.T
lambda_ = 0
for i in range(0,50):
    #AX
    X = mul(A,X)
    #print(X)
    lambda_ = abs(X[2])
    #print(lambda_)
    X = X/lambda_    
print("Maximum Eigen value : ",float(lambda_))

#3b
#INVERSE POWER METHOD
#3x3 matrix
import numpy as np
A = np.zeros((3,3))
#print("Enter the values of 3x3 matrix : ")
"""
for i in range(0,3):
    for j in range(0,3):
        A[i][j] = float(input())
"""
A[0][0] = 1
A[0][1] = 2
A[0][2] = 3
A[1][0] = 0
A[1][1] = 1
A[1][2] = 4 
A[2][0] = 5
A[2][1] = 6
A[2][2] = 0

X_initial = np.array([[1,1,1]])
#X_initial.shape[0]

def mul(A,B):
    result = np.zeros((3,1))
    #print(A,B,result)
    for i in range(0,A.shape[0]):
        for j in range(0,B.shape[1]):
            for k in range(0,B.shape[0]):
                result[i][j] += A[i][k]*B[k][j]
    #print("Result : ",result)
    return result

X = X_initial.T
stop_value = 0.0010
B = np.linalg.inv(A)   #B is inverse of A
#print(B)
lambda_ = 0
stop_cond = 1
while(stop_cond > stop_value):
    #BX
    X = mul(B,X)
    #print(X)
    stop_cond = lambda_
    lambda_ = abs(X[0])
    #print(lambda_)
    X = X/lambda_ 
    stop_cond = abs(stop_cond - lambda_)
    #print(stop_cond)
print("Minimum Eigen value : ",1/lambda_)


#Q4
#RREF and REF USING LIBRARY FUNCTIONS
import numpy as np
from sympy import *
x,y,z = symbols('x y z')
e1 = 1*x+0*y+1*z-3
e2 = 2*x+3*y+4*z-7
e3 = -1*x-3*y-3*z+4
a = [[e1.coeff(x),e1.coeff(y),e1.coeff(z)],
     [e2.coeff(x),e2.coeff(y),e2.coeff(z)],
     [e3.coeff(x),e3.coeff(y),e3.coeff(z)]]
ab=Matrix([[e1.coeff(x),e1.coeff(y),e1.coeff(z),-(e1-a[0][0]*x-a[0][1]*y-a[0][2]*z)],
     [e2.coeff(x),e2.coeff(y),e2.coeff(z),-(e2-a[1][0]*x-a[1][1]*y-a[1][2]*z)],
     [e3.coeff(x),e3.coeff(y),e3.coeff(z),-(e3-a[2][0]*x-a[2][1]*y-a[2][2]*z)]])     #augmented matrix = ab
print("Matrix :{}".format(ab))

M_rref=ab.rref()
print(M_rref[0])
#print("The RREF of the Matrix is :{}".format(M_rref))

#4b
#Finding Solution of Equation using Library
# we use AX=B
# So X=A inverse*B
import numpy as np
from sympy import *
x,y,z = symbols('x y z')
e1 = -9*x+4*y+4*z-2
e2 = -8*x+3*y+4*z-4
e3 = -16*x+8*y+7*z-6
A = np.array([[e1.coeff(x),e1.coeff(y),e1.coeff(z)],
     [e2.coeff(x),e2.coeff(y),e2.coeff(z)],
     [e3.coeff(x),e3.coeff(y),e3.coeff(z)]],dtype = 'float64')
B=np.array([-(e1-A[0][0]*x-A[0][1]*y-A[0][2]*z),-(e2-A[1][0]*x-A[1][1]*y-A[1][2]*z),-(e3-A[2][0]*x-A[2][1]*y-A[2][2]*z)],dtype = 'float64')
#print(A,B)
#print(np.linalg.inv(A))
X=np.linalg.inv(A).dot(B)
print(X[0],X[1],X[2])


#4 3.Diagnolizing a Matrix using library
import numpy as np
from sympy import *
M=Matrix([[1,-3,3],[3,-5,3],[6,-6,4]])
print("Matrix :{}".format(M))
P,D=M.diagonalize()
print("Diagonal of Matrix :{}".format(D))



#4 4.Canonical form - Diagnolizing a Matrix using library
import numpy as np
from sympy import *

A = np.zeros((3,3))

x1 = var('x1')
x2 = var('x2')
x3 = var('x3')
#x1_x2,x1_x3,x1_x2,x2_x3,x1_x3,x2_x3 = symbols('x1x2 x1x3 x1x2 x2x3 x1x3 x2x3')
equation =(-9*x1**2 + 4*x1*x2 + 4*x1*x3 - 8*x1*x2 + 3*x2**2 + 4*x2*x3 - 16*x1*x3 + 8*x2*x3 + 7*x3**2)

def canonicalToMatrix(eq):
    A[0][0] = eq.coeff(x1**2)
    A[0][1] = 0.5*eq.coeff(x1*x2)
    A[0][2] = 0.5*eq.coeff(x1*x3)
    A[1][0] = 0.5*eq.coeff(x1*x2)
    A[1][1] = eq.coeff(x2**2)
    A[1][2] = 0.5*eq.coeff(x2*x3)
    A[2][0] = 0.5*eq.coeff(x1*x3)
    A[2][1] = 0.5*eq.coeff(x2*x3)
    A[2][2] = eq.coeff(x3**2)

canonicalToMatrix(equation)
print("Matrix :{}".format(M))
P,D=M.diagonalize()
print("Diagonal of Matrix :{}".format(D))
D = np.array(D)

print("%0.2f"%D[0][0]," x1**2 + ","%0.2f"%D[1][1]," x2**2 + ","%0.2f"%D[2][2]," x3**2")


#Q5
import numpy as np
from sympy import *
#5a
#DIAGONALIZATION
#l = lambda
A = np.zeros((3,3))
#print("Enter the values of 3x3 matrix : ")
"""
for i in range(0,3):
    for j in range(0,3):
        A[i][j] = float(input())
"""
"""
A[0][0] = -9
A[0][1] = 4
A[0][2] = 4
A[1][0] = -8
A[1][1] = 3
A[1][2] = 4
A[2][0] = -16
A[2][1] = 8
A[2][2] = 7
"""

#x1,x2,x3 = symbols('x1 x2 x3')
x1 = var('x1')
x2 = var('x2')
x3 = var('x3')
#x1_x2,x1_x3,x1_x2,x2_x3,x1_x3,x2_x3 = symbols('x1x2 x1x3 x1x2 x2x3 x1x3 x2x3')
equation =sympify(-9*x1**2 + 4*x1*x2 + 4*x1*x3 - 8*x1*x2 + 3*x2**2 + 4*x2*x3 - 16*x1*x3 + 8*x2*x3 + 7*x3**2)
print("Equation : ")
print("---------")
print(equation)

def canonicalToMatrix(eq):
    A[0][0] = eq.coeff(x1**2)
    A[0][1] = 0.5*eq.coeff(x1*x2)
    A[0][2] = 0.5*eq.coeff(x1*x3)
    A[1][0] = 0.5*eq.coeff(x1*x2)
    A[1][1] = eq.coeff(x2**2)
    A[1][2] = 0.5*eq.coeff(x2*x3)
    A[2][0] = 0.5*eq.coeff(x1*x3)
    A[2][1] = 0.5*eq.coeff(x2*x3)
    A[2][2] = eq.coeff(x3**2)

canonicalToMatrix(equation)
print(A)


def trace(A):
    return A[0][0]+A[1][1]+A[2][2]
def det(A):
    return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])) - (A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]))+(A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
def sum_minors(A):
    return (A[1][1]*A[2][2]-A[1][2]*A[2][1]) + (A[0][0]*A[2][2]-A[2][0]*A[0][2]) + (A[0][0]*A[1][1]-A[1][0]*A[0][1])

def getRREF(mat):                               #To get RREF of a REF matrix
    mat.reverse()
    for i in mat:
        if(1 in i):
            ones=i.index(1)
        else:
            continue
        ind=mat.index(i)
        for j in range(ind,len(mat)-1):
            key=mat[j+1][ones]
            mat[j+1]=subRows(mat[j+1],mat[ind], key)
    mat.reverse()
    return mat

x,y,z = symbols('x y z')
I = np.identity(3)

det_ = det(A)
trace_=trace(A)
sum_ = sum_minors(A)

eqn = x**3 - x**2 *trace_ + x*sum_ - det_
sol = solve(eqn)
print("Eigen values : ")
print(sol)

X = np.array([[x,y,z]])
X=X.T

print("Eigen vectors : " )
for i in sol:
    # i =lambdas
    mat = list(A-i*I)
    #a = Matrix(A-i*I).rref()[0]
    a  = Matrix(getRREF(mat))
    e = a*X
    print(solve((e[0],e[1],e[2])))

#eig_val = (np.linalg.eig(A)[0])
eig_vec = (np.linalg.eig(A)[1])
d = ((Matrix(np.linalg.inv(eig_vec)) * Matrix(A) *eig_vec))
d =np.array(d)
print("Diagonal Matrix : ")
print("----------------")
for i in range(0,3):
    for j in range(0,3):
        print("%0.2f"%d[i][j],end= "    ")
    print("")
print("Result : ")
print("-------")
print("%0.2f"%d[0][0]," x1**2 + ","%0.2f"%d[1][1]," x2**2 + ","%0.2f"%d[2][2]," x3**2")


#5b Gauss Jordan Elimination
import numpy as np
from sympy import *

#n=int(input("enter the number of unknowns"))
n = 3
x=np.zeros(n)

"""
for i in range(n):
  for j in range(n+1):
    a[i][j]=float(input("a["+str(i)+"]"+"["+str(j)+"]="))
"""
x,y,z = symbols('x y z')
"""
#unique solution
e1 = -9*x+4*y+4*z-2
e2 = -8*x+3*y+4*z-4
e3 = -16*x+8*y+7*z-6
"""
"""
e1 = 1*x-3*y+1*z-4
e2 = -1*x+2*y-5*z-3
e3 = 5*x-13*y+13*z-8
"""
"""
#infinite solution
e1 = 2*x+2*y+2*z+2
e2 = 2*x+3*y+2*z-4
e3 = 1*x+1*y+1*z+1
"""
e1 = 1*x-1*y+3*z-3
e2 = -2*x+2*y-6*z-6
e3 = 0*x+1*y-5*z+3

a = [[e1.coeff(x),e1.coeff(y),e1.coeff(z)],
     [e2.coeff(x),e2.coeff(y),e2.coeff(z)],
     [e3.coeff(x),e3.coeff(y),e3.coeff(z)]]
ab = [[e1.coeff(x),e1.coeff(y),e1.coeff(z),-(e1-a[0][0]*x-a[0][1]*y-a[0][2]*z)],
     [e2.coeff(x),e2.coeff(y),e2.coeff(z),-(e2-a[1][0]*x-a[1][1]*y-a[1][2]*z)],
     [e3.coeff(x),e3.coeff(y),e3.coeff(z),-(e3-a[2][0]*x-a[2][1]*y-a[2][2]*z)]]
print("Augumented Matrix : ")
print("-------------------")
print(ab)

def getRREF(mat):                               #To get RREF of a REF matrix
    mat.reverse()
    for i in mat:
        if(1 in i):
            ones=i.index(1)
        else:
            continue
        ind=mat.index(i)
        for j in range(ind,len(mat)-1):
            key=mat[j+1][ones]
            mat[j+1]=subRows(mat[j+1],mat[ind], key)
    mat.reverse()
    return mat
def subRows(l1,l2,key):
    ans=[]
    for i in range(len(l1)):

        ans.append(l1[i]-(key*l2[i]))
       
    return ans
def getRowEcholon(mat):                         #To get REF of the matrix
    ptr=0
    mat = list(mat)
    for i in mat:
        #print(i)
        div=i[ptr]
       
        if(div!=0):
                for k in range(ptr,len(i)):
                    i[k]=i[k]/div
        ind=mat.index(i)
        for j in range(ind,len(mat)-1):
            key=mat[j+1][ptr]
            mat[j+1]=subRows(mat[j+1],mat[ind],key)
        ptr+=1
    return mat

def getRank(mat):
    mat =getRREF(mat)
    rank=0                              #computes rank by counting num of non zero rows in row echolon form
    for i in mat:
        temp=0
        for j in range(len(i)):
            if(i[j]==0):
                temp+=1
        if(temp==len(i)):
            rank+=1
    return(len(mat)-rank)

"""
for i in range(n):
  x[i]=a[i][n]/a[i][i]

print("The Solution is :")
for i in range(n):
  print(x[i])
"""
unknowns = 3
rankAB=getRank(ab)
rankA=getRank(a)
print("")
print("Rank of matrix [A:B] : ",rankAB)
print("Rank of matrix A : ",rankA,"\n")
if (rankAB==rankA):
    print("Consistent")
    if(rankAB ==unknowns):
        print("Unique Solutions : ")
        print("------------------")
        result=getRREF(getRowEcholon(ab))
        #print(result)
        for i in range(unknowns):
            print("Value of variable ",i+1," : ",result[i][-1])
    elif(rankAB<unknowns):
        print("Infinite solutions ")
elif(rankA<rankAB):
    print("Inconsistent : No solutions")





