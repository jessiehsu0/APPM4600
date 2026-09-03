"""
 This program is a warm up for coding. You get used to the coding 
format and practice some coding skills. 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 


import numpy as np 
import numpy.linalg as la
import math
import time

def driver():

     n = 100
     x = np.linspace(0,np.pi,n)

# this is a function handle.  You can use it to define 
# functions instead of using a subroutine like you 
# have to in a true low level language.     
     f = lambda x: x**2 + 4*x + 2*np.exp(x)
     g = lambda x: 6*x**3 + 2*np.sin(x)

     y = f(x)
     w = g(x)

     a = np.array([0,1])
     b = np.array([1,0])
     n2 = 2

     A = np.array([[2, 1, 0], [3, 2, 1], [1, 1, 4]])
     x = np.array([4, 2, 1])
     nMatrix = 3
     mMatrix = 3

# evaluate the dot product of y and w    
     start_time_1 = time.perf_counter() 
     dp = dotProduct(y,w,n)
     end_time_1 = time.perf_counter()

     dp2 = dotProduct(a,b,n2)

     start_time_2 = time.perf_counter() 
     dpNumpy = np.dot(y, w)
     end_time_2 = time.perf_counter() 

     start_time_3 = time.perf_counter() 
     Ax = matrixVectorMultiply(A, x, nMatrix, mMatrix)
     end_time_3 = time.perf_counter() 

     start_time_4 = time.perf_counter() 
     AxNumpy = np.dot(A, x)
     end_time_4 = time.perf_counter()

# print the output
     print('the dot product is : ', dp)
     print('the numpy dot product : ', dpNumpy)
     print('the second dot product is : ', dp2)
     print('the matrix-vector product is : ', Ax )
     print('the numpy matrix-vector product is : ', AxNumpy)
     print(f"time to complete dot product: {end_time_1-start_time_1:.6f} seconds")
     print(f"time to complete numpy dot product: {end_time_2-start_time_2:.6f} seconds")
     print(f"time to complete matrix-vector product: {end_time_3-start_time_3:.6f} seconds")
     print(f"time to complete numpy matrix-vector product: {end_time_4-start_time_4:.6f} seconds")

     return
     
def dotProduct(x,y,n):
#   Computes the dot product of the n x 1 vectors x and y
     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  

def matrixVectorMultiply(A, x, n, m):
# computes matrix vector multiplication between n x m matrix A and m x 1 vector x
     v = np.zeros(n)
     for i in range(n):
          for j in range(m):
               v[i] += A[i][j] * x[j]

     return v
     
driver()               
