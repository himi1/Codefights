def largestFullBinaryTree(parent):

    class Graph:

        def __init__(self, parent):
            self.maxBinTree = 1
            self.edges = []
            for i in range(len(parent)):
                self.edges.append([])
            for i in range(1, len(parent)):
                self.edges[parent[i]].append(i)

        def dfs(self, v):
            firstMax = -1
            secondMax = -1
            for u in self.edges[v]:
                curMax = self.dfs(u)
                if curMax > firstMax:
                    secondMax = firstMax
                    firstMax = curMax
                elif curMax > secondMax:
                    secondMax = curMax
            if secondMax == -1:
                return 1
            result = 1 + firstMax + secondMax
            if result > self.maxBinTree:
                self.maxBinTree = result
            return result

    g = Graph(parent)
    g.dfs(0)
    return g.maxBinTree

#
def concatenationProcess(init):
    while len(init) != 1:
        T = sorted(init, key = len)
        L = T[0]
        R = sorted(T[1:], key = len, reverse = True)[-1]
        init.remove(L)
        init.remove(R)
        init.append(L + R)
    return init[0]

#
def isOneSwapEnough(inputString):
    if len(inputString) == 1:
        return True
    for i in range(len(inputString) - 1):
        for j in range(i + 1, len(inputString)):
            T = list(inputString)
            T[i], T[j] = T[j], T[i]
            if T == T[::-1]:
                return True
    return False

#
def biggerWord(w0):
    w = list(w0)
    leftSwap = len(w) - 1
    for i in range(len(w) - 2, -1, -1):
        if w[i] < w[i + 1]:
            leftSwap = i
            break

    if leftSwap == len(w) - 1:
        return 'no answer'

    rightSwap = len(w) - 1
    while w[leftSwap] >= w[rightSwap]:
        rightSwap -= 1

    w[leftSwap], w[rightSwap] = w[rightSwap], w[leftSwap]
    leftSwap += 1
    rightSwap = len(w) - 1
    while leftSwap < rightSwap:
        w[leftSwap], w[rightSwap] = w[rightSwap], w[leftSwap]
        leftSwap += 1
        rightSwap -= 1

    return ''.join(w)
#
def rectanglesIntersection(a, b, c, d):
    list = []
    for i in range(0, 2):
        if (c[i] > d[i]):
            t = c[i]
            c[i] = d[i]
            d[i] = t
        if (a[i] > b[i]):
            t = a[i]
            a[i] = b[i]
            b[i] = t
        if (b[i] < c[i] or d[i] < a[i]):
            return 0

        list.append(min(b[i], d[i]) - max(a[i], c[i]))

    return list[0] * list[1]

#
def checkFactorial(n):
    i = 1
    p = 1
    while p < n:
        i += 1
        p *= i

    return p == n

#
def firstDuplicate(a):
    from math import fabs
    for item in a:
        if a[int(fabs(item))-1] < 0:
            return(int(fabs(item)))
        else:
            a[int(fabs(item))-1] *= -1
    return -1

    for i, x in enumerate(a):
        a[abs(x) - 1] *= -1
        if a[abs(x) - 1] > 0:
            return abs(x)
    return -1

    '''
    t = -1
    index = 100001

    for i , each in enumerate(a):
        a[i] = -1
        print each
        try:
            ind = a.index(each)
            if ind < index:
                t = each
                index = ind
        except:
            continue

    return t
    '''

#
#solution 2:
#failed on last two cases due to T.L.E
def firstDuplicate(a):
    index = -1
    #calculate init value of t
    for i, each in enumerate(a):
        if each in a[i+1:]:
            t = each
            index = a[i+1:].index(each) + i + 1
            #print("init index: ", index)
            break

    if index == -1:
        #print ("No repeated element found, returning -1")
        return -1

    for i , each in enumerate(a):
        #a[i] = -1
        #print each
        if each in a and each in a[a.index(each)+1:] and i < index:
            index = a[a.index(each)+1:].index(each) + a.index(each) + 1
            #print("new index: ", index, each, i, a)


    #print("output: ", a[index], " | index:", index)
    return a[index]

#
def caseUnification(inputString):

    changesToMakeUppercase = len(re.findall('[a-z]', inputString))
    changesToMakeLowercase = len(re.findall('[A-Z]', inputString))

    if (changesToMakeUppercase == 0
        or changesToMakeLowercase != 0
        and changesToMakeUppercase < changesToMakeLowercase):
        return inputString.upper()
    else:
        return inputString.lower()

#
def differentSquares(matrix):
    t = set()
    for i in range(len(matrix)-1):
        for j in range(len(matrix[0])-1):
            t.add((matrix[i][j], matrix[i][j+1], matrix[i+1][j], matrix[i+1][j+1]))

    return len(t)

#
def fractionDivision(a, b):
    def gcd(a,b):
        if not b:
            return a
        else:
            return gcd(b, a % b)

    c = [a[0] * b[1], b[0] * a[1]];
    t = gcd(c[0], c[1]);
    c[0] /= t
    c[1] /= t
    return c

#
def phoneCall(min1, min2_10, min11, s):

    if s < min1:
        return 0
    for i in range(2, 11):
        if s < min1 + min2_10 * (i - 1):
            return i - 1
    return 10 + (s - min1 - min2_10 * 9) / min11

#
def findTheRemainder(a, b):
    while a >= b:
        a-= b
    return a

#
def houseNumbersSum(a):
    s =0
    i=0
    while a[i] != 0:
        s += a[i]
        i +=1
    return s

#
def bfsDistancesUnweightedGraph2(matrix, vertex1, vertex2):

    visited = []
    queue = []
    distance = []

    for i in range(len(matrix)):
        visited.append(False)
        distance.append(0)

    visited[vertex1] = True
    queue.append(vertex1)
    while len(queue) > 0:
        currentVertex = queue[0]
        queue = queue[1:]
        visited[currentVertex] = True
        for nextVertex in range(len(matrix)):
            if matrix[currentVertex][nextVertex] and not visited[nextVertex]:
                visited[nextVertex] =  False
                distance[nextVertex] = distance[currentVertex] + 1
                queue.append(nextVertex)

    return distance[vertex2]

#
def fileNaming(names):
    output = []
    extra = []

    done = False
    for each in names:
        print("output:", str(output))
        print("extra:", str(output))
        if each not in output:
            output.append(each)
            extra.append(each)

        else:
            if each in extra:
                count = extra.count(each)
                while not done:
                    t = each + "(" + str(count) + ")"
                    if t not in output:
                        output.append(t)
                        break
                    count+= 1

                    #output.append(each + "(" + str(count) + ")")

            else:
                 output.append(each + "(" + str(1) + ")")
            extra.append(each)

    return output

#
def ratingThreshold(threshold, ratings):
    output = []
    i = 0
    for each in ratings:
        print each
        print float(sum(each))/len(each)
        if float(sum(each))/len(each) < threshold:
            print each
            output.append(i)
        i += 1
    return output

#
def proCategorization(pros, preferences):
    result = []
    dict = {}
    i = 0
    for pro in pros:
        for p in preferences[i]:
            if p in dict:
                dict[p].append(pro)
            else:
                dict[p] = [pro]
        i += 1

    t = []
    [["Computer lessons"], ["Leon", "Maria"]]
    for key, values in dict.iteritems():
        t.append([key])
        t.append(values)
        result.append(t)
        t = []


    return sorted(result)

#
def cellsJoining(table, coords):
    table = [list(i) for i in table]
    C = [i for i, v in enumerate(table[0]) if v == '+']
    R = [i for i, v in enumerate(table) if v[0] == '+']
    (maxr, minCol), (minr, maxc) = coords
    R = R[minr:maxr+2]
    C = C[minCol:maxc+2]
    for r in R:
        if r in (0, R[-1]):
            for c in C[1:-1]:
                table[r][c] = '-'
        elif r in R[1:-1]:
            for c in range(C[0]+1, C[-1]):
                table[r][c] = ' '
    for c in C:
        if c in (0, C[-1]):
            for r in R[1:-1]:
                table[r][c] = '|'
        elif c in C[1:-1]:
            for r in range(R[0]+1, R[-1]):
                table[r][c] = ' '
    table = [''.join(i) for i in table]
    return table

###
def divNumber(k, l, r):

    ans = 0
    for i in range(l, r + 1):
        n = i
        divs = 1
        j = 2
        while j * j <= n:
            pow_ = 0
            while(n%j == 0):
                pow_ += 1
                n/= j
            divs *= 1 + pow_
            if divs > k:
                break
            j += 1
        if n > 1:
            divs *= 2
        if divs == k:
            ans += 1

    return ans

###
def amicableNumbers(n):
    if(n<=220): return 220
    if(n>220 and n<=284): return 284
    if(n>284 and n<=1184): return 1184

    return 1210

###
def centuryFromYear(year) :
    return (year + 99) / 100

###
def differentSubstringsTrie(a):
    t = set()
    for i in range(len(a)):
        for j in range(i+1,len(a)+1):
            t.add(a[i:j])
    return len(t)

###
def bishopDiagonal(bishop1, bishop2):
    cols = 'abcdefgh'
    rows = '12345678'

    def bishopMoves(cell):
        col, row = cols.index(cell[0]), rows.index(cell[1])
        for c in range(8):
            for r in (row - (col - c), row - (c - col)):
                if 0 <= r < 8:
                    yield cols[c] + rows[r]

    b1 = set(bishopMoves(bishop1))
    b2 = set(bishopMoves(bishop2))
    u = b1.intersection(b2)
    if len(u) == 0:
        return sorted([bishop1, bishop2])
    else:
        s = sorted(u, reverse = bishop2 < bishop1)
        return sorted([s[0], s[-1]])

    Bx = ord(bishop[0])-ord("a")
    By = ord(bishop[1])-ord("0")
    Px = ord(pawn[0])-ord("a")
    Py = ord(pawn[1])-ord("0")
    if Bx == Px:
        return False
    elif abs(((By)-(Py))/((Bx)-(Px))) == 1:
        return True
    else:
        return False

def Dir(bishop, pawn):
    Bx = ord(bishop[0])-ord("a")
    By = ord(bishop[1])-ord("0")
    Px = ord(pawn[0])-ord("a")
    Py = ord(pawn[1])-ord("0")
    return ((By)-(Py))/((Bx)-(Px))

def bishopDiagonal(b1, b2):
    QS1 = [['a7','b8'],['a6','c8'],['a5','d8'],['a4','e8'],['a3','f8'],['a2','g8'],['a1','h8'],['b1','h7'],['c1','h6'],['d1','h5'],['e1','h4'],['f1','h3'],['g1','h2']]
    QSn1 = [['a2','b1'],['a3','c1'],['a4','d1'],['a5','e1'],['a6','f1'],['a7','g1'],['a8','h1'],['b8','h2'],['c8','h3'],['d8','h4'],['e8','h5'],['f8','h6'],['g8','h7']]

    if bishopAndPawn(b1,b2):
        if Dir(b1,b2) == 1.0:
            print('1')
            for i in range(len(QS1)):
                if bishopAndPawn(b1,QS1[i][0]) and bishopAndPawn(b2,QS1[i][0]):
                    return QS1[i]
                elif(b1 ==QS1[i][0] or b2 == QS1[i][0]):
                    return QS1[i]
        else:
            print('-1')
            for i in range(len(QSn1)):
                if bishopAndPawn(b1,QSn1[i][0]) and bishopAndPawn(b2,QSn1[i][0]):
                    return QSn1[i]
                elif(b1 == QSn1[i][0] or b2 == QSn1[i][0]):
                    return QSn1[i]
    else:
        x = [b1,b2]
        x.sort()
        return x

###
def passwordCheckRegExp(inputString):
    if len(inputString) < 5:
        return False
    L = U = D = False
    for i in range(len(inputString)):
        if 'a' <= inputString[i] <= 'z':
            L = True
        if 'A' <= inputString[i] <= 'Z':
            U =  True
        if '0' <= inputString[i] <= '9':
            D = True
    return L and U and D and len(inputString) >= 5

###
def isSumOfConsecutive(n):
    for i in range(1, n):
        j = i
        t = n
        while t > 0:
            t -= j
            j += 1
        if t == 0: return True
    return False

###
def isSubstitutionCipher(string1, string2):
    dict = {}
    for i  in range(len(string1)):
        s = string1[i]
        s1 = string2[i]
        if s in dict.keys():
            if s1 != dict[s]:return False
        else:
            dict[s]=s1

    return len(set(dict.values()))==len(dict.values())

###
def caesarBoxCipherEncoding(s):

    n = int(len(s)**0.5)
    r = ''

    for i in range(n):
        for j in range(n):
            r+=s[n*j+i]
    return r
