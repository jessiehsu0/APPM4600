"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
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


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time

def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     for N in [100, 500, 1000, 2000, 4000, 5000]:
          ''' Right hand side'''
          b = np.random.rand(N,1)
          A = np.random.rand(N,N)

          print(f"N = {N}")
          start = time.time()
          x_solve = scila.solve(A,b)
          end = time.time()
          test_solve = np.matmul(A,x_solve)
          r = la.norm(test_solve-b)

          elapsed = end - start
     
          # print("scila solve error: ", r)
          print("scila solve time: ", elapsed)

          start = time.time()
          lu_and_piv = scila.lu_factor(A)
          end_lu_factor = time.time()
          x_lu_solve = scila.lu_solve(lu_and_piv, b)
          end_lu_solve = time.time()
          test_lu_solve = np.matmul(A, x_lu_solve)
          r_2 = la.norm(test_lu_solve - b)

          elapsed_lu_factor = end_lu_factor - start
          elapsed_lu_solve = end_lu_solve - end_lu_factor

          # print("scila lu factor and solve error: ", r_2)
          print("scila lu factorization time: ", elapsed_lu_factor)
          print("scila lu solve time: ", elapsed_lu_solve)
          print("scila lu factor + solve time (total): ", elapsed_lu_solve + elapsed_lu_factor)

          start = time.time()
          x_np_solve = la.solve(A, b)
          end = time.time()
          test_np_solve = np.matmul(A, x_np_solve)
          r_3 = la.norm(test_np_solve-b)

          elapsed = end - start

          # print("numpy solve error: ", r_3)
          print("numpy solve time: ", elapsed)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)



     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
