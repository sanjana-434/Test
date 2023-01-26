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

#INVERSE POWER METHOD
#3x3 matrix
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