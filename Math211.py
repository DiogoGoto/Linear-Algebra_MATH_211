import math
from operator import le
#Example Matrix format
A = [
     [-1,1,4],
     [4,4,0],
     [-3,-5,-2]
     ]

B = [
    [1,0,0],
     [0,1,0],
     [0,0,1]
     ]

X=[
    [2],
    [4],
    [3]
]
#removes the ith row and jth collum
def minor(A,i,j):
    i-=1
    j-=1
    A_ = []
    for k in range(len(A)):
        A_r =[]
        A_r = A[k].copy()
        A_r.pop(j)
        A_.append(A_r)
    A_.pop(i)
    return determinant(A_)

def cof(A,i,j):
    return ((-1)**(i+j)) * (minor(A,i,j))
           

def determinant(A):
    result = 0

    if len(A) == 2 and len(A[0]) == 2: # if matrix is 2 x 2
        return (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])

    
    for j in range(len(A[0])):
        
        result += A[0][j] * cof(A,1,j+1) # sum of a1,j * (-1)^(1+j) * minor(A)1,j
    return result

def cof_matrix(A):
    B=[]
    for i in range(len(A)):
        Br =[]
        for j in range(len(A[1])):

            Br.append(cof(A,i+1,j+1))
        B.append(Br)
    return B


def transpose(A):
    At=[]
    for i in range(len(A)):
        Ar=[]
        for j in range(len(A[0])):
            Ar.append(A[j][i])
        At.append(Ar)
    return At


def adjugate(A):
    return transpose(cof_matrix(A))


def inverse(A):
    #Work in progress
    print("Work in progress")    


def sum(A,B):
    if not(len(A) == len(B) and len(A[0]) == len(B[0])):
        return 'A and B are no the same size'
    
    C = []
    for i in range(len(A)):
        Cr = []
        for j in range(len(A[i])):
            Cr.append(A[i][j])
        C.append(Cr)

    for i in range(len(A)): 
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]

    return C

def scalar_mul(A,k):
    A_ = []
    for i in range(len(A)):
        A_r = []
        for j in range(len(A[i])):
            A_r.append(A[i][j])
        A_.append(A_r)
    

    for i in range(len(A_)):
        for j in range(len(A_[i])):
            A_[i][j] *= k
    return A_

def subtraction(A,B):
    return sum(A,scalar_mul(B,-1))
    

def matrix_prod(A,B):
    C = []

    for i in range(len(A)):
        Cr =[]
        for j in range(len(B[i])):
            Cr.append(0)
        C.append(Cr)


    for i in range(len(B[0])):
        Bc=[]
        for j in range(len(B)):
            Bc.append(B[j][i])
        
        for j in range(len(A[i])):
            for k in range(len(Bc)):
                C[k][i] += A[k][j] * Bc[j]
    
    return C

def inverse(A):
    return scalar_mul(adjugate(A),1/determinant(A))

def cramers_rule(A,B):
    deta = determinant(A)
    if deta == 0:
        return "inf many solutions/ no solution (det(A) = 0)"
    if len(A) != len(B):
        return 'B need to vector collum with the same number of rows of A'

    X = []

    for i in range(len(B)):
        Ai =[]
        for j in range(len(A)):
            AiR = A[j].copy()
            Ai.append(AiR)

        for j in range(len(A[i])):
            Ai[j][i] = B[j][0]
        
        deti = determinant(Ai)
        X.append([deti/deta])


    return X

def print_det(A):
    counter=1
    for i in A:
        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print("A = ", end="")
            counter = 1000
        else:
            print("    ", end="")
        print(i)
        counter += 1

    print()
    print("det(A) = ",determinant(A))

def print_minor(A,i_,j_):
    counter=1
    i_ -= 1
    j_ -= 1

    for i in A:
        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print("A = ", end="")
            counter = 1000
        else:
            print("    ", end="")
        print(i)
        counter += 1
    print()
    print("minor(A)",i_+1,j_+1, "= ",minor(A,i_,j_))

def print_cofmatrix(A_):
    counter=1

    A = cof_matrix(A_)

    for i in A:
        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print("Cofactor Matrix of  A = ", end="")
            counter = 1000
        else:
            print("                        ", end="")
        print(i)
        counter += 1


def print_transpose(A):
    counter=1

    C = transpose(A)

    for i in C:
        if (((len(C) // 2) == counter) and (len(C)%2 ==0) ) or ((len(C)%2 !=0) and (((math.ceil(len(C) / 2))) == counter)):
            print("Transpose of A = ", end="")
            counter = 1000
        else:
            print("                 ", end="")
        print(i)
        counter += 1

def print_adjugate(A_):
    counter=1

    C = adjugate(A_)

    for i in C:
        if (((len(C) // 2) == counter) and (len(C)%2 ==0) ) or ((len(C)%2 !=0) and (((math.ceil(len(C) / 2))) == counter)):
            print("Adjugate of A = ", end="")
            counter = 1000
        else:
            print("                ", end="")
        print(i)
        counter += 1    

def print_scarlar_mul(A_,k):
    counter=1

    C = scalar_mul(A_,k)

    for i in C:
        if (((len(C) // 2) == counter) and (len(C)%2 ==0) ) or ((len(C)%2 !=0) and (((math.ceil(len(C) / 2))) == counter)):
            print(k,"* A  = ", end="")
            counter = 1000
        else:
            print("          ", end="")
        print(i)
        counter += 1        



def biggest_row(A):
    index = 2
    jindex = 1
    R1 = ''
    R2 = ''
    for j in range(len(A[0])):
        R1 += str(A[0][j])
        R2 += str(A[1][j])

    if len(R1) < len(R2):
        jindex = 1

    for _ in range(len(A)-1):    
        while len(R1) >= len(R2) and index <= (len(A)-1):
            R2 = ''
            for i in range(len(A[index])):
                R2 += str(A[index][i])
            index += 1

        if len(R1) < len(R2):
            jindex = index
            R1 = R2
    

    return [jindex-1,R1]

def print_sum(A,B):
    counter = 1

    C = sum(A,B)

    sizeA = biggest_row(A)
    sizeB = biggest_row(B)
    

    for i in range(len(A)):
        Ri = ''
        RBi = ''
        for k in range(len(A[i])):
            Ri += str(A[i][k])
        for k in range(len(B[i])):
            RBi += str(B[i][k])

        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( " +  " , end="")
            print(B[i], end="")
            if sizeB[0]+1 != counter:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print(" = ", end="")

            print(C[i])

            counter = 1000

        else:
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( "    " , end="") 
            print(B[i], end="")


            if sizeB[0] != i:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print( "   " , end="") 
            print(C[i])
        counter += 1

def print_matrix_prod(A,B):
    counter = 1

    C = matrix_prod(A,B)

    sizeA = biggest_row(A)
    sizeB = biggest_row(B)
    

    for i in range(len(A)):
        Ri = ''
        RBi = ''
        for k in range(len(A[i])):
            Ri += str(A[i][k])
        for k in range(len(B[i])):
            RBi += str(B[i][k])

        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( " *  " , end="")
            print(B[i], end="")
            if sizeB[0]+1 != counter:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print(" = ", end="")

            print(C[i])

            counter = 1000

        else:
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( "    " , end="") 
            print(B[i], end="")


            if sizeB[0] != i:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print( "   " , end="") 
            print(C[i])
        counter += 1    

def print_inverse(A_):
    counter=1

    C = inverse(A_)

    for i in C:
        if (((len(C) // 2) == counter) and (len(C)%2 ==0) ) or ((len(C)%2 !=0) and (((math.ceil(len(C) / 2))) == counter)):
            print("Inverse of A = ", end="")
            counter = 1000
        else:
            print("               ", end="")
        print(i)
        counter += 1

def print_subtraction(A,B):
    counter = 1

    C = subtraction(A,B)

    sizeA = biggest_row(A)
    sizeB = biggest_row(B)
    

    for i in range(len(A)):
        Ri = ''
        RBi = ''
        for k in range(len(A[i])):
            Ri += str(A[i][k])
        for k in range(len(B[i])):
            RBi += str(B[i][k])

        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( " -  " , end="")
            print(B[i], end="")
            if sizeB[0]+1 != counter:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print(" = ", end="")

            print(C[i])

            counter = 1000

        else:
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( "    " , end="") 
            print(B[i], end="")


            if sizeB[0] != i:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print( "   " , end="") 
            print(C[i])
        counter += 1

def print_cramers(A,B_):
    counter = 1

    if B_ == 'inf many solutions/ no solution (det(A) = 0)':
        print(B_)
        return
    
    if B_ == "B need to vector collum with the same number of rows of A":
        print(B_)  
        return

    B = cramers_rule(A,B_)

    C = matrix_prod(A,B)

    sizeA = biggest_row(A)
    sizeB = biggest_row(B)

    print("Cramer's Rule ")
    print("A * X = B")
    print("the values for the variables are in Matrix X")
    print()

    for i in range(len(A)):
        Ri = ''
        RBi = ''
        for k in range(len(A[i])):
            Ri += str(A[i][k])
        for k in range(len(B[i])):
            RBi += str(B[i][k])

        if (((len(A) // 2) == counter) and (len(A)%2 ==0) ) or ((len(A)%2 !=0) and (((math.ceil(len(A) / 2))) == counter)):
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( " *  " , end="")
            print(B[i], end="")
            if sizeB[0]+1 != counter:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print(" = ", end="")

            print(C[i])

            counter = 1000

        else:
            print(A[i], end="")
            if sizeA[0]+1 != counter:
                for _ in range(len(sizeA[1])-len(Ri)):
                    print(" ",end="")
            print( "    " , end="") 
            print(B[i], end="")


            if sizeB[0] != i:
                for _ in range(len(sizeB[1])-len(RBi)):
                    print(" ",end="")
            print( "   " , end="") 
            print(C[i])
        counter += 1

    print()
      

print_minor(A,3,1)
print()
print_det(A)
print()
print_cofmatrix(A)
print()
print_transpose(A)
print()
print_adjugate(A)
print()
print_scarlar_mul(A,-1)
print()
print_sum(A,B)
print()
print_matrix_prod(A,B)
print()
print_inverse(A)
print()
print_cramers(A,X)