###
def caseUnification(inputString):

    changesToMakeUppercase = len(re.findall('[a-z]', inputString))
    changesToMakeLowercase = len(re.findall('[A-Z]', inputString))

    if (changesToMakeUppercase == 0
        or changesToMakeLowercase != 0
        and changesToMakeUppercase < changesToMakeLowercase):
        return inputString.upper()
    else:
        return inputString.lower()

###
def equidistantTriples(coordinates):

    ans = 0
    for i in range(1, len(coordinates)):
        left = i - 1
        right = i + 1
        while left >= 0 and right < len(coordinates):
            distL = coordinates[i] - coordinates[left]
            distR =  coordinates[right] - coordinates[i]
            if distL == distR:
                ans += 1
                left -= 1
                right += 1
            elif distL < distR:
                left -= 1
            else:
                right += 1

    return ans

###
def caesarBoxCipherEncoding(input):
    l = int(len(input)**(0.5))
    output = ''

    for i in range(l):
        for j in range(l):
            output+=input[i + l*j]
    return output

###
def sumBelowBound(bound):

    left = 0
    right = bound + 1
    while right - left > 1:
        middle = (right + left) / 2
        if middle * (middle + 1) / 2 <= bound:
            left = middle
        else:
            right = middle

    return left

###
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

###
def toAndFro(a, b, t):

    length = abs(b - a)
    t %= 2 * length
    if t <= length:
        return  a+(b - a)/ abs(b - a) * t
    else:
        t -= length
        return b + (a - b) / abs(a - b) * t

###
def firstNotDivisible(divisors, start):
    while True:
        if all(start % x for x in divisors):
            return start
        start+= 1

###
def toDecimal(k, n):

    result = 0
    power = 1
    for i in range(len(n) - 1, -1, -1):
        result += int(n[i]) * power
        power *= k

    return result
###
def isEarlier(time1, time2):
    if  time1 < time2 :
        return True
    return False

###
def newNumeralSystem(number):
    s = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = s.index(number)
    i = 0
    t = ["A" + " + " + number]
    i = 1
    counter = n/2 if n %2 == 0 else n/2 + 1
    while i < counter:
        x = n - i
        t.append(s[i+1] + " + " +  s[x])
        i+= 1

    return t

###
def digitsProduct(product):

    answerDigits = []
    answer = 0

    if product == 0:
        return 10

    if product == 1:
        return 1

    for divisor in range(9, 1, -1):
        while product % divisor == 0:
            product /= divisor
            answerDigits.append(divisor)

    if product > 1:
        return -1


    for i in range(0, len(answerDigits)):
        answer = answer + answerDigits[i]*10**i
    return answer

###
def bfsComponentSize(matrix):
    visited = [False for i in range(len(matrix))]
    queue = []
    componentSize = 0

    visited[1] = True
    queue.append(1)
    while len(queue) > 0:
        currentVertex = queue.pop()
        visited[currentVertex] = True
        componentSize += 1
        for nextVertex in range(len(matrix)):
            if  matrix[currentVertex][nextVertex] and not visited[nextVertex]:
                visited[nextVertex] = True
                queue.append(nextVertex)

    return componentSize

###
def increaseNumberRoundness(n):

    gotToSignificant = False
    while n > 0:
        if n % 10 == 0 and gotToSignificant:
            return True
        elif n % 10 != 0:
            gotToSignificant = True
        n /= 10

    return False

###
def maxDigit(n):

    result = 0
    while n > 0:
        result = max(result, n % 10)
        n /= 10

    return result

###
def partialSort(input, k):
    answer = []
    infinity = 10 ** 9

    for i in range(k):
        index = 0
        j = 0
        for j in range(len(input)):
            if input[j] < input[index]:
                index = j
        answer.append(input[index])
        input[index] = infinity
    for i in range(len(input)):
        if input[i] != infinity:
            answer.append(input[i])

    return answer

###
def primeFactors(n):
    f = []
    d = 2

    while n >= 2:
        if n % d == 0:
            f.append(d)
            n /= d
        else:
            d += 1
    return f

###
def chessBoardCellColor(cell1, cell2):

    def getX(pos):
        return ord(pos[0]) - ord('A')
    def getY(pos):
        return ord(pos[0]) - ord('1')

    sum1 = getX(cell1[0]) + getY(cell1[1])
    sum2 = getX(cell2[0]) + getY(cell2[1])
    if sum1%2 == sum2%2:
        return True
    return False

###
def isDivisibleBy6(inputString):

    digitSum = 0
    leftBound = ord('0')
    rightBound = ord('9')
    answer = []
    mask = list(inputString)
    asteriskPos = -1

    for i in range(len(mask)):
        if (leftBound <= ord(mask[i]) and
          ord(mask[i]) <= rightBound):
            digitSum += ord(mask[i]) - leftBound
        else:
            asteriskPos = i

    for i in range(10):
        if (digitSum + i) % 3 == 0:
            mask[asteriskPos] = chr(leftBound + i)
            if  (int(mask[len(mask) - 1] - leftBound) % 2 == 0 :
                answer.append(''.join(mask))

    return answer

###
def sequencePeakElement(sequence):
    left = 1
    right = len(sequence) - 2
    while left < right:
        middle = (left + right) / 2
        if sequence[middle] > max(sequence[middle - 1], sequence[middle + 1]):
            left = right = middle
            break
        if sequence[middle - 1] < sequence[middle]:
            left = middle + 1
        else:
            right = middle - 1
    return sequence[left]
