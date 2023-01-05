import Matrices as la
import math
import sympy

def included_angle(A,B):
    return math.degrees(math.acos((dot_product(A,B))/(lenth(A)*lenth(B))))

def dot_product(A,B):
    sum = 0
    for i in range(len(A)):
        sum += A[i] * B[i]
    return sum

def cross_product(A,B):
    dvectors = {}
    for i in range(len(A)):
        r = []
        for _ in range(len(A)):
            r.append(0)
        dvectors[i] = r.copy()
        dvectors[i][i] = 1
    
    C = []
    for i in range(len(A)):
        r = []
        for _ in range(len(A)):
            r.append(0)
        C.append(r.copy())

    for i in range(len(A)):
        C[0][i] = (f"{i}")
        C[1][i] = A[i]
        C[2][i] = B[i]

    cofterms = det_cross_prod(C)
    for i in cofterms:
        dvectors[int(i)] = vector_scalar_mul(dvectors[int(i)],cofterms[i])
    
    result = []
    for i in range(len(dvectors[0])):
        result.append(0)

    for i in dvectors:
        result = vector_sum(result,dvectors[i])
    return result
   
def boxproduct(A,B,C):
    return dot_product(A,cross_product(B,C))

def intersection_lines(A,B):
    parametric_eq = {}
    for i in range(len(A[0])):
        parametric_eq[f"A{i}"] = str(A[0][i]) + '+' + f"{A[2][i]}*{A[1]}"
    for i in range(len(B[0])):
        parametric_eq[f"B{i}"] = str(B[0][i]) + '+' + f"{B[2][i]}*{B[1]}"

    Amatrix = []
    for _ in range(len(B[2])):
        Amr = []
        for _ in range(2):
            Amr.append(0)
        Amatrix.append(Amr)
   
    X = []
    for i,j in enumerate(parametric_eq):
        if A[1] in parametric_eq[j]:
            Amatrix[i][0] = -int(parametric_eq[j][parametric_eq[j].index('+')+1:parametric_eq[j].index('*')])
            X.append(int(parametric_eq[j][:parametric_eq[j].index('+')]))
        if B[1] in parametric_eq[j]:
            Amatrix[i-3][1] = int(parametric_eq[j][parametric_eq[j].index('+')+1:parametric_eq[j].index('*')])
            X[i-3] -= int(parametric_eq[j][:parametric_eq[j].index('+')])  
    for i, var in enumerate(X):
        Amatrix[i].append(var)
    for i, val in enumerate(X):
        X[i] = [val]

    Amatrix[0] = vector_scalar_mul(Amatrix[0],1/Amatrix[0][0])
    for i in range(1,len(Amatrix[0])):
        Amatrix[i] = vector_sum(Amatrix[i],vector_scalar_mul(Amatrix[0],-Amatrix[i][0]))


    for i in range(1,len(Amatrix[0])):
        if Amatrix[i][1] == 0:
            return "L1 and L2 don't intersect"
        Amatrix[i] = vector_scalar_mul(Amatrix[i],1/Amatrix[i][1])
    
    delete=[]
    for i,row in enumerate(Amatrix):
        if row == Amatrix[i-1]:
            delete.append(i)
        X[i][0] = Amatrix[i][-1]     
    for i in delete:
        for j in range(len(Amatrix)):
            Amatrix[j].pop(-1)
        Amatrix.pop(i)
        X.pop(i)
    var_values = la.cramers_rule(Amatrix,X)

    intesectA = []
    intesectB = []

    for i,j in enumerate(parametric_eq):
        if 'A' in j:
           intesectA.append(int(parametric_eq[j][:parametric_eq[j].index('+')]) + int(parametric_eq[j][parametric_eq[j].index('+')+1:-2]) * var_values[0][0])
        if 'B' in j:
            intesectB.append(int(parametric_eq[j][:parametric_eq[j].index('+')]) + int(parametric_eq[j][parametric_eq[j].index('+')+1:-2]) * var_values[1][0])

    #if intesectA == intesectB:
    return intesectA, var_values

def det_cross_prod(A):

    if len(A) == 2 and len(A[0]) == 2: # if matrix is 2 x 2
        return (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])

    coftemrs = {}

    for j in range(len(A[0])):
        coftemrs[f"{j}"] =  la.cof(A,1,j+1)  # sum of a1,j * (-1)^(1+j) * minor(A)1,j
    
    return coftemrs

def vector_scalar_mul(A,k):
    A_ = A.copy()
    for i in range(len(A_)):
        A_[i] *= k
        A_[i] = float(f"{A_[i]}")
    return A_

def vector_sum(A,B):    
    C = list(range(len(A)))

    for i in range(len(A)): 
        C[i] = float(f"{A[i] + B[i]}")

    return C

def vector(A,B):    
    C = list(range(len(A)))

    for i in range(len(A)):
        C[i] = float(f"{B[i] - A[i]:.4f}")

    return C

def lenth(A):
    return math.sqrt(sum([i**2 for i in A ]))

def projection(vector_1,vector_2):
    return vector_scalar_mul(vector_2,dot_product(vector_1,vector_2) / (lenth(vector_2)**2))
    
def closest_point(P,line_point,d_vector):
    Pline_P = vector(line_point,P)
    Pline_Q = projection(Pline_P,d_vector)
    Q = vector_sum(line_point,Pline_Q)
    return Q, lenth(vector(Q,P))

def area_paralelogram(P1,P2,P3):
    return lenth(cross_product(vector(P1,P2),vector(P2,P3))),lenth(cross_product(vector(P1,P2),vector(P2,P3)))**2

def area_triangle(P1,P2,P3):
    return lenth(cross_product(vector(P1,P2),vector(P2,P3))) / 2,(lenth(cross_product(vector(P1,P2),vector(P2,P3))) / 2) ** 2

def intersection_planes(normal_1, ans_1, normal_2, ans_2):
    direction_vector = cross_product(normal_1, normal_2)
    eq1 = normal_1.copy()
    eq1.append(ans_1)
    eq2 = normal_2.copy()
    eq2.append(ans_2)

    eq = sympy.Matrix([eq1,eq2]).rref()

    x = ''
    y = ''
    z = ''

    if eq[1][0] != 0:
        x = 't'
    if eq[1][1] != 1:
        y = 't'
    if eq [1][1] != 2 or eq[1][0] == 2:
        z='t'
   
    for index ,entry in enumerate(eq[0]):

        if 0 <= index <= 3:
            if (entry == 0 or index == 0) or x == 't':
                continue
            else:
                if index == 3:
                    if entry > 0:
                        x += '+' + str(entry)
                    else:
                        x +=  str(entry)
                else:
                    if entry < 0:
                        x += '-' + str(entry) + 't'
                    else:
                        x += str(entry) + 't'

        if 4 <= index <= 7:
            if (entry == 0 or index == 5) or y == 't':
                continue
            else:
                if index == 7:
                    if entry < 0:
                        y += str(entry)
                    else:
                        y += '+' + str(entry)
                else:
                    if entry > 0:
                        y += '-' + str(entry) + 't'
                    else:
                        y += str(entry) + 't'
    for index ,entry in enumerate(eq[0]):
        if 4 <= index <= 7:
            if (entry == 0 or index == 6) or z == 't':
                continue
            else:
                if index == 7:
                    print(entry)
                    if entry < 0:
                        z += str(entry)
                    else:
                        z += '+' + str(entry)

                else:
                    print(entry)
                    if entry < 0:
                        z += '-' + str(entry) + 't'
                    else:
                        z += str(entry) + 't'

    if x == 't':
        x = 0
        if 't' in y and y[y.index('t')+1:] != '':
            y = float(y[y.index('t')+1:])
        elif y != '':
            if 't' not in y:
                y = float(y)
            else:
                y = 0
        if 't' in z and z[z.index('t')+1:] != '':
            z = float(z[z.index('t')+1:])
        elif z != '' :
            if 't' not in z:
                z = float(z)
            else:
                z=0
    elif y == 't':
        print(x)
        if 't' in x and x[x.index('t')+1:] != '':
            x = float(x[x.index('t')+1:])
        elif x != '':
            if 't' not in x:
                x = float(x)
            else:
                x=0
        y = 0
        if 't' in z and z[z.index('t')+1:] != '':
            z = float(z[z.index('t')+1:])
        elif z != '' :
            if 't' not in z:
                z = float(z)
            else:
                z = 0
    elif z == 't':
        if 't' in x and x[x.index('t')+1:] != '':
            x = float(x[x.index('t')+1:])
        elif x != '':
            if 't' not in x:
                x = float(x)
            else:
                x = 0
        if 't' in y and y[y.index('t')+1:] != '':
            y = float(y[y.index('t')+1:])
        elif y != '':
            if 't' not in y:
                y = float(y)
            else:
                y = 0
            
        z = 0
    
    return [[x,y,z],'t',direction_vector]

def closest_dist_non_parallel(P1,d1,P2,d2):
    parametric_eq = {}
    for i in range(3):
        if P1[i] >= 0:
            parametric_eq[f'A{i}'] = f'{d1[i]}t+{P1[i]}'
        else:
            parametric_eq[f'A{i}'] = f'{d1[i]}t{P1[i]}'
        if P2[i] >= 0:
            parametric_eq[f'B{i}'] = f'{d2[i]}s+{P2[i]}'
        else:
            parametric_eq[f'B{i}'] = f'{d2[i]}s{P2[i]}'

    for i in range(3):
        D1i =  float(parametric_eq[f'A{i}'][:parametric_eq[f'A{i}'].index('t')])
        D2i = -float(parametric_eq[f'B{i}'][:parametric_eq[f'B{i}'].index('s')])
        independet_term = float(parametric_eq[f'A{i}'][parametric_eq[f'A{i}'].index('t')+1:]) - float(parametric_eq[f'B{i}'][parametric_eq[f'B{i}'].index('s')+1:])

        parametric_eq[f'C{i}'] = vector_scalar_mul([D1i,D2i,-independet_term],d1[i])
        parametric_eq[f'D{i}'] = vector_scalar_mul([D1i,D2i,-independet_term],d2[i])

    for i in range(1,3):
        parametric_eq[f'C{0}'] = vector_sum(parametric_eq[f'C{0}'],parametric_eq[f'C{i}'])
        parametric_eq[f'D{0}'] = vector_sum(parametric_eq[f'D{0}'],parametric_eq[f'D{i}'])
    

    system = sympy.Matrix([parametric_eq[f'C{0}'],parametric_eq[f'D{0}']]).rref()

    t = system[0][2]
    s = system[0][5]

    for eq in parametric_eq:
        if 'A' in eq:
            parametric_eq[eq] = float(parametric_eq[eq][:parametric_eq[eq].index('t')]) * t + float(parametric_eq[eq][parametric_eq[eq].index('t')+1:])
        if 'B' in eq:
            parametric_eq[eq] = float(parametric_eq[eq][:parametric_eq[eq].index('s')]) * s + float(parametric_eq[eq][parametric_eq[eq].index('s')+1:])
   
    Q1 = []
    Q2 = []
    for i in range(3):
        Q1.append(parametric_eq[f'A{i}'])
        Q2.append(parametric_eq[f'B{i}'])
     
    distance = lenth(vector(Q1,Q2))

    return Q1,Q2,distance,(t,s)

def plane_eq(P,P1,d):
    normal = cross_product(vector(P1,P),d)
    eq = str(normal[0]) + 'x'
    if normal[1] >=0:
        eq += '+' + str(normal[1]) + 'y'
    else:
        eq += str(normal[1]) + 'y'
    if normal[2] >= 0:
        eq += '+' + str(normal[2]) + 'z='
    else:
        eq += str(normal[2]) + 'z='
    return eq+ str(dot_product(normal,P1))

def plane_line_intersection(normal,const,P1,d):
    parametric_eq ={}
    for eq in range(3):
        parametric_eq[eq] = vector_scalar_mul([P1[eq],d[eq]],normal[eq])

    const = -sum([parametric_eq[eq][0] for eq in parametric_eq]) + const
    t = sum([parametric_eq[eq][-1] for eq in parametric_eq]) 
    
    if t == 0:
        return 'line is in the plane or does not intersect'
    
    t = const/t

    instersection = vector_sum(P1,vector_scalar_mul(d,t))

    
    return instersection, t

def rhombus(PA,P1,d,M):
    AM = vector(PA,M)
    C = vector_sum(M,AM)
    B = [[P1[i],d[i]] for i in range(len(P1))]
    BM = []
    for entry in B:
        BM.append(entry.copy())
    
    for index,entry in enumerate(M):
        BM[index][0] *= -1
        BM[index][0] += entry
        BM[index][1] *= -1
    
    for index, entry in enumerate(AM):
        BM[index][0] *= entry
        BM[index][1] *= entry
    const = 0
    t = 0
    
    for entry in BM:
        const += entry[0]
        t += entry[1]

    t = -const/t

    for index in range(len(B)):
        B[index] = B[index][0] + (B[index][1] * t)

    BM = vector(B,M)
    D = vector_sum(BM, M)
    return B,C,D

#Sample Vector
P1 = [-1,1,-1]
P2 = [-1,-1,2]
P3 = [-3,3,2]

A = [0,-1,0]
B = [-1,-1,-1]
C = [-1,-1,-2]
D = [-2,-2,2]

Q1 = [11.75,-7.25, 7.5]
Q2 = [29.725, 26.575,-2.55]
d = [0,0,3]

#Sample line equations
L1 = [[11,4,-12],'t',[6,6,-9]]
L2 =[[26,18,11],'s',[1,2,1]]

N1 = [-2,-5,-5]
ans1 = -6
N2 = [2,-1,2]
ans2 = 5

d1 = [-1,0,-2]
d2 = [-2,2,-3]

if __name__ == '__main__':
    print(f'Boxproduct({[round(a,4) for a in A]},{[round(a,4) for a in B]},{[round(a,4) for a in C]}) =', boxproduct(A,B,C))
    print()
    print(f'The Lines{L1} and {L2} intersect{intersection_lines(L1,L2)[0]}\n and the variable are {intersection_lines(L1,L2)[1]}')
    print()
    print(f'Dot_product({[round(a,4) for a in A]},{[round(a,4) for a in B]} =',dot_product(A,B))
    print(f'Cos(\u03B8) =',dot_product(A,B)/(lenth(A)*lenth(B)))
    print(f'\u03B8 =',included_angle(A,B))
    print()
    print(f'Closest point btw {P1} and [{P2},+t,{d}] is {closest_point(P1,P2,d)[0]} and the distance is {closest_point(P1,P2,d)[1]}')
    print()
    print(f'Cross_product({[round(a,4) for a in A]},{[round(a,4) for a in B]}) = {cross_product(A,B)}')
    print()
    #print(f'The planes {N1}={ans1} and {N2}={ans2} , intersect',intersection_planes(N1,ans1,N2,ans2))
    print()
    print(f'The area of the triangle defined {P1}, {P2}, {P3}',area_triangle(P1,P2,P3))
    print()
    print(f'The Closest dist btw the lines [{P1},+t,{d1}] and [{P2},+t{d2}] is {closest_dist_non_parallel(P1,d1,P2,d2)[-2]}\n',
            closest_dist_non_parallel(P1,d1,P2,d2))
    print()
    print(f'The plane defined by the points{P1},{P2},{P3},',plane_eq(P1,P2,P3))
    print()
    print(f'the plane {N1}={ans1} and the line [{P1}+t{d}] intersect ',plane_line_intersection(N1,ans1,P1,d))
    print()
    print(rhombus([-2,-1,2],[-2,-1,2],[-6,0,18],[2,1,1]))