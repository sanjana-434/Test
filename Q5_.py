
import numpy as np
from sympy import *
#Q5
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




