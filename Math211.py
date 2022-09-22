import math
#Example Matrix format
A=[[3,4,-3],
   [-3,5,6],
   [3,5,-1]]

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

def cof_matrix(A):
    B=[]
    for i in range(len(A)):
        Br =[]
        for j in range(len(A[1])):
            Br.append(cof(A,i+1,j+1))
        B.append(Br)
    return B

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

def print_cofmatrix(A_,):
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


print_minor(A,2,1)
print()
print_det(A)
print()
print_cofmatrix(A)
