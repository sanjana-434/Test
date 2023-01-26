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
