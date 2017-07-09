###
def rightTriangle(sides):
    sides = sorted(sides)
    if (sides[0]**2 + sides[1]**2 == sides[2]**2): return True
    return False

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
            if matrix[currentVertex][nextVertex] and not visited[nextVertex]:
                visited[nextVertex] = True
                queue.append(nextVertex)

    return componentSize

###
def waterTubes(water, flowPerMinute):
    result = 0

    for i in range(len(water)):
        minutes = water[i] / flowPerMinute[i]
        if water[i] % flowPerMinute[i] != 0:
            minutes += 1

        if result < minutes:
            result = minutes
    return result

###
def isEarlier(time1, time2):
     if (time1[0] * 60 + time1[1] < time2[0] * 60 + time2[1]): return True

     return False

###
def dfsComponentSize(matrix, vertex):

    def dfs(currentVertex, visited):
        visited[currentVertex] = True
        componentSize = 1
        for nextVertex in range(len(matrix)):
            if matrix[currentVertex][nextVertex] and not visited[nextVertex]:
                componentSize += dfs(nextVertex, visited)
        return componentSize

    visited = []

    for i in range(len(matrix)):
        visited.append(False)

    componentSize = dfs(vertex, visited)

    return componentSize

###
def quickSort(a, l, r):

    if l >= r:
        return a

    x = a[l]
    i = l
    j = r

    while i <= j:
        while a[i] < x:
            i += 1
        while a[j] > x:
            j -= 1
        if i <= j:
            t = a[i]
            a[i] = a[j]
            a[j] = t
            i += 1
            j -= 1

    quickSort(a, l, j)
    quickSort(a, i, r)

    return a

###
def isIPv4Address(inputString):
    s = inputString.split('.')
    if len(s) != 4: return False
    for each in s:
        if (len(each)==0) or (not each.isdigit()) or (not 0 <= int(each)<= 255):
            return False
    return True

###
def maxSubarray(inputArray):
        currentMax = 0
        result = 0

        for i in range(len(inputArray)):
                currentMax = max(0, currentMax + inputArray[i])
                result = max(result, currentMax)

        return result

###
def greatestCommonPrimeDivisor(a, b):

    gcd = -1
    divisor = 2
    while a > 1 and b > 1:
        if a % divisor == 0 and b % divisor == 0:
            gcd =  divisor
        while a % divisor == 0:
            a /= divisor
        while b % divisor == 0:
            b /= divisor
        divisor += 1
    return gcd

###
def arrayMaximalAdjacentDifference(inputArray):
        max = 0
        for i in range(1, len(inputArray)):
                if abs(inputArray[i] - inputArray[i - 1]) > max:
                        max = abs(inputArray[i] - inputArray[i - 1])

        return max

###
def variableName(name):

    for i in range(len(name)):
        if (not ('a' <= name[i] and name[i] <= 'z' or
                    'A' <= name[i] and name[i] <= 'Z' or
                    '0' <= name[i] and name[i] <= '9' or
                    name[i] == '_')):
            return False

    if '0' <= name[0] and name[0] <= '9':
        return False

    return True

###
def tennisSet(score1, score2):
    if score1 < score2:
        tmp = score1
        score1 =  score2
        score2 = tmp
    if score1 == 6 and score2 < 5 or score1 == 7 and score2 < 7 and score2 >= 5:
        return True
    return False

###
def leastCommonPrimeDivisor(a, b):
    i = 2
    while i <= a and i <= b:
        if (a % i == 0 and b % i == 0):
            return i
        i += 1

    return -1

###
def isCaseInsensitivePalindrome(inputString):

    for i in range(len(inputString) / 2):
        c = [
                inputString[i],
                inputString[len(inputString) - i - 1]
        ]
        for j in range(2):
            if c[j] >= 'a' and c[j] <= 'z':
                c[j] = chr(
                        ord(c[j]) - ord('a') + ord('A')
                )
        if c[0] != c[1]:
            return False

    return True

###
def maxZeros(n):
    answer = 0
    maxZeros = 0
    for k in range(2, n + 1):
        numZeros = 0
        value = n
        while value:
            if value % k == 0:
                numZeros += 1
            value /= k
        if numZeros > maxZeros:
            maxZeros = numZeros
            answer = k
    return answer

###
def fullName(first, last):
    return first + " " + last

###
def factorialTrailingZeros(n):
    result = 0
    for i in range(5, n + 1, 5):
        number = i
        while number % 5 == 0:
            number /= 5
            result += 1
    return result

###
def permutationShift(permutation):
    minShift = 0
    maxShift = 0
    for i in range(len(permutation)):
        if permutation[i] - i > maxShift:
            maxShift =  permutation[i] - i
        if permutation[i] - i < minShift:
            minShift = permutation[i] - i
    return maxShift - minShift

###
def passwordCheckRegExp(inputString):
    t1 = len(re.findall(r'([A-Z])', inputString))
    t2 = len(re.findall(r'([a-z])', inputString))
    t3 = len(re.findall(r'([0-9])', inputString))

    if (len(inputString) >= 5) and t1 != 0 and t2 != 0 and t3 != 0:
        return True

    return False

###
def factorialsProductTrailingZeros(l, r):
    result = 0
    last = 0
    for i in range(1, r + 1):
        number = i
        while number % 5 == 0:
            number /= 5
            last += 1
        if i >= l:
            result += last
    return result

###
def isSumOfConsecutive(n):
    for start in range(1, n):
        number = n
        subtrahend = start
        while number > 0:
            number -= subtrahend
            subtrahend += 1
        if number == 0:
            return  True
    return False

###
def returnSecondParameter(a, b):
    return b

###
def largestDistance(a):
    mx = [a[0], a[1]]
    mn = [a[0], a[1]]
    for i in range(len(a)):
        k = i % 2
        if a[i] > mx[k]:
            mx[k] = a[i]
        elif a[i] < mn[k]:
            mn[k] = a[i]
    return max(abs(mx[0]-mn[0]), abs(mx[1]-mn[1]))

###
def applesDistribution(apples, boxCapacity, maxResidue):
    t = 0
    for i in range(1, boxCapacity + 1):
        if apples % i <= maxResidue:
            t += 1
    return t
