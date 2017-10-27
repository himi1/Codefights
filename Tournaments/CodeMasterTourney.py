###
def countIncreasingSequences(n, k):
    dp = []
    ans = 0

    for i in range(n + 1):
        line = []
        for j in range(k + 1):
            line.append(0)
        dp.append(line)
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for q in range(j):
                dp[i][j] += dp[i - 1][q]

    for j in range(1, k + 1):
        ans += dp[n][j]

    return ans

###
def maxDivisor(left, right, divisor):

    i = right
    while i >= left:
        if i % divisor == 0:
            return i
        i -= 1
    return  -1

###
def eulersTotientFunction(n):
    div = 2
    r = n

    while div * div <= n:
        if n % div == 0:
            while n % div == 0:
                n /= div
            r -= r / div
        div += 1
    if n > 1:
        r -= r / n

    return r

###
def candles(candlesNumber, makeNew):
    burned = 0
    leftovers = 0
    while candlesNumber > 0:
        burned += candlesNumber
        leftovers += candlesNumber
        candlesNumber = leftovers / makeNew
        leftovers %= makeNew
    return burned

###
def longestWord(text):
    answer = ''
    current = []

    for i in range(len(text)):
        if ('a' <= text[i] and text[i] <= 'z'
                or 'A' <= text[i] and text[i] <= 'Z'):
            current.append(text[i])
            if len(current) > len(answer):
                answer =  "".join(current)
        else:
            current = []

    return answer

###
def isDigit(symbol):
    return symbol.isdigit()

###
def crossingSum(matrix, row, column):

    result = 0
    for i in range(len(matrix)):
        result += matrix[i][column]
    for i in range(len(matrix[0])):
        result += matrix[row][i]
    result -= matrix[row][column]

    return result

###
def digitDistanceNumber(n):
    result = 0
    lastDigit = n % 10
    tenPower = 1
    n /= 10
    while n:
        result += tenPower * abs(n % 10 - lastDigit)
        tenPower *= 10
        lastDigit = n % 10
        n /= 10

    return result

###
def isInfiniteProcess(a, b):
    while (a != b):
        a+= 1
        b-= 1
        if (a == b): return False
        if (a > b): return True

    return False

###
def isDivisibleBy3(inputString):
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
            answer.append(''.join(mask))

    return answer

###
def gcdNaive(a, b):

    gcd = 1
    for divisor in range(2, min(a, b) + 1):
        if a % divisor == 0 and b % divisor == 0:
            gcd =  divisor

    return gcd

###
def permutationShift(permutation):
    min = max = 0
    for i in range(len(permutation)):
        if (permutation[i] - i > max):
            max = permutation[i] - i
        if (permutation[i] - i < min):
            min = permutation[i] - i

    return max - min

###
def hangman(word, letters):

    neededLetters = [False] * 26
    need = 0
    for i in range(len(word)):
        c = ord(word[i]) - ord('a')
        if not neededLetters[c]:
            neededLetters[c] = True
            need += 1

    missed = 0
    for i in range(len(letters)):
        if not (missed <= 6 and need > 0):
            break
        c = ord(letters[i]) - ord('a')
        if neededLetters[c]:
            neededLetters[c] = False
            need -= 1
        else:
            missed += 1

    return need == 0

###
def truncateString(s):

    def truncate(l, r):
        if (l >= r): return ""
        newL = l
        newR = r
        left = ord(s[l]) - ord('0')
        right = ord(s[r - 1]) - ord('0')
        if left % 3 == 0:
            newL += 1
        elif right % 3 == 0:
            newR -= 1
        elif (left + right) % 3 == 0:
            newL += 1
            newR -= 1
        else:
            return s[l : r]

        return truncate(newL, newR)

    return truncate(0, len(s))

###
def nontransitiveDice(dice):
    count = len(dice)
    l = len(dice[0])
    list = [[0,0],[0,0],[0,0]]

    for i in range(count - 1):
        for j in range(i+1, count):
            for k in range(l):
                for h in range(l):
                    if( dice[i][k] > dice[j][h]): list[i+j-1][0] += 1
                    else: list[i+j-1][1] += 1

    for i in range(len(list)):
        if(list[i][0] >= list[i][1]):
            list[i] = 1
        else:
            list[i] = 0

        temp = "".join(str(x) for x in list)
    if (temp != '101' and temp != '010'):
        return False
    return True

###
def arrayMinimumIndex(inputArray):

    indexOfMinimum = 0
    for i in range(1, len(inputArray)):
        if inputArray[i] < inputArray[indexOfMinimum]:
            indexOfMinimum = i
    return indexOfMinimum

###
def factorizedGCD(a, b):
    j = 0
    result = 1
    for i in range(len(a)):
        while j < len(b) and a[i] > b[j]:
            j += 1
        if  j < len(b) and a[i] == b[j] :
            result *= a[i]
            j += 1
    return result

###
def halvingSum(n):
    sum = 0
    i = n
    while i > 0:
        sum += i
        i /= 2
    return sum

###
def stringsCrossover(inputArray, result):

    answer = 0

    for i in range(len(inputArray)):
        for j in range(i + 1, len(inputArray)):
            correct = True
            for k in range(len(result)):
                if (result[k] != inputArray[i][k]
                  and result[k] != inputArray[j][k]):
                    correct = False
                    break
            if correct:
                answer += 1
    return answer

###
def hailstoneSequence(n):
    t = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        t += 1
    return t

###
def fibonacciIndex(n):

  a = 0
  b = 1
  i = 0
  while len(str(a)) < n:
    c = a + b
    a = b
    b = c
    i += 1

  return i

###
 def fractionDivision(a, b):

    def gcdEuclid(a, b):
        if a == 0:
            return b
        return gcdEuclid(b % a, b)

    c = [a[0] * b[1], a[1] * b[0]]
    gcd = gcdEuclid(c[0], c[1])

    c[0] /= gcd
    c[1] /= gcd

    return c

###
def maxSubarray(inputArray):
        currentMax = 0
        result = 0

        for i in range(len(inputArray)):
                currentMax = max(0, currentMax + inputArray[i])
                result = max(result, currentMax)

        return result

###
def piecesOfDistinctLengths(strawLength):
    i = 1
    while ((i + 1) * (i + 2) / 2 <= strawLength):
        i += 1

    return i

###
def piecesOfDistinctLengths(strawLength):
    t = 1
    s = 0
    while s + t <= strawLength:
        s += t
        t += 1
    return t - 1

###
def checkIncreasingSequence(seq):

    for i in range(1, len(seq)):
        if seq[i] <= seq[i + 1]:
            return False

    return True

###
def bettingGame(l):

    s = 0
    for i in range(len(l)):
        s += l[i]
    if s == 0:
        return False

    return  s % len(l) == 0

###
def isInformationConsistent(evidences):
    for i in range(len(evidences[0])):
        t = 0
        for j in range(len(evidences)):
            if evidences[j][i] * t < 0:
                return False
            else:
                t += evidences[j][i]
    return True

###
def applesDistribution(apples, boxCapacity, maxResidue):
    t = 0
    for i in range(1, boxCapacity + 1):
        if apples % i <= maxResidue:
            t += 1
    return t

###
def twoArraysNthElement(array1, array2, n):

    def lowerBound(array, elem):
        l = -1
        r = len(array)
        while l + 1 < r:
            mid = (l + r) / 2
            if array[mid] <= elem:
                l = mid
            else:
                r = mid
        return l

    l = -1
    r = len(array1)

    while l + 1 < r:
        mid = (l + r) / 2
        pos = lowerBound(array2, array1[mid])

        if mid + pos + 1 <= n:
            l = mid
        else:
            r = mid

    if l > -1 and l + lowerBound(array2, array1[l]) + 1 == n:
        return array1[l]
    return twoArraysNthElement(array2, array1, n)

###
def isSuspiciousRespondent(ans1, ans2, ans3):
     return  (ans1 and ans2 and ans3) or (not ans1 and not ans2 and not ans3)

###
def fractionSum(a, b):

    def gcdEuclid(a, b):
        if a == 0:
            return b
        return gcdEuclid(b % a, a)

    c = [a[0] * b[1] + a[1] * b[0], a[1] * b[1]]
    gcd = gcdEuclid(a[1], b[1])

    c[0] /= gcd
    c[1] /= gcd

    return c

###
def arePrizesOK(first, second, third):
    if first < second:
        return  False
    if second < third:
        return False
    return True

###
def howManySundays(n, startDay):
    dict = {"Monday" : 1, "Tuesday" : 2, "Wednesday":3, "Thursday" : 4, "Friday" : 5,"Saturday":6, "Sunday":0}

    t = n + dict[startDay]
    return t/7

###
def chartFix(chart):
    toRemove = []
    for i in range(len(chart)):
        cur = i
        for j in range(i):
            if chart[j] < chart[i]:
                cur = max(cur, toRemove[j] + i - j - 1)
        toRemove.append(cur)
    res = float('inf')
    for i in range(len(toRemove)):
        res = min(res, toRemove[len(toRemove) - i - 1] + i)
    return res

###
def rectanglesIntersection(a, b, c, d):

    intersection = []

    for i in range(2):
        if a[i] > b[i]:
            t = a[i]
            a[i] = b[i]
            b[i] = t
        if c[i] > d[i]:
            t = c[i]
            c[i] = d[i]
            d[i] = t
        if b[i] < c[i] or d[i] < a[i]:
            return 0
        intersection += [min(b[i], d[i]) - max(a[i], c[i])]

    return  intersection[0] * intersection[1]

###
def charactersRearrangement(string1, string2):
    string1 = sorted(string1)
    string2 = sorted(string2)
    return string1==string2

###
def isCorrectSentence(inputString):

    leadChar = inputString[0]
    endChar = inputString[len(inputString) - 1]

    if ('A' <= leadChar and leadChar <= 'Z'
          and endChar == '.'):
        return True
    else:
        return False

###
def digitCharactersSum(ch1, ch2):
    x1 = ord(ch1) - ord('0')
    x2 = ord(ch2) - ord('0')
    if x1 + x2 < 10:
        return chr(ord('0') + x1 + x2)
    else:
        return '1' + chr(ord('0') + (x1 + x2) % 10)

###
def isSmooth(arr):
    l = len(arr)
    if l%2 == 0: t = arr[l/2]+arr[l/2-1]
    else: t = arr[l//2]

    if arr[0]==t and arr[-1]==t:
        return True
    return False

###
def factorizedGCD(a, b):
    j = 0
    result = 1
    for i in range(len(a)):
        while j < len(b) and a[i] > b[j]:
            j += 1
        if j < len(b) and a[i] == b[j]:
            result *= a[i]
            j += 1
    return result

###
def videoPart(part, total):

    def getSeconds(time):
        h = int(time[0:2])
        m = int(time[3:5])
        s = int(time[6:8])
        return h * 60 * 60 + m * 60 + s

    def gcd(a, b):
        while a > 0:
            tmp = a
            a = b % a
            b = tmp
        return b

    partTime = getSeconds(part)
    totalTime = getSeconds(total)
    divisor = gcd(partTime, totalTime)
    return [partTime / divisor, totalTime / divisor]

###
def sumOfPowers(n, divisor):
    s = 0
    for i in range(1, n+1):
        t = 1
        count = 0
        while True:
            t *= divisor
            if (i % t != 0):
                break
            count+= 1
        s+= count
    return s

###
def isPangram(sentence):
    found = []
    result = True
    for i in range(26):
        found.append(False)
    for i in range(len(sentence)):
        code = ord(sentence[i])
        if ord('A') <= code and code <= ord('Z'):
            code += ord('a') - ord('A')
        if ord('a') <= code and code <= ord('z'):
            found[code - ord('a')] =  True

    for i in range(26):
        if not found[i]:
            result = False

    return result

###
def arithmeticProgression(element1, element2, n):
    return 0 if (element2 < element1) else (element1 + (n - 1) * abs(element1 - element2))

###
def noAdjacentBits(a):

    lastBit = 0
    idx = 0
    while (1 << idx) <= a:
        curBit = (a >> idx) & 1
        if lastBit == 1 and curBit == 1:
            return False
        lastBit = curBit
        idx += 1

    return True

###
def extractEachKth(inputArray, k):

    result = []
    for i in range(len(inputArray)):
        if  (i + 1) %k != 0 :
            result.append(inputArray[i])
    return result

###
def zFunctionNaive(s):
    t = []
    for i in range(len(s)):
        c = 0
        for a, b in zip(s[i:], s):
            if a == b: c += 1
            else: break
        t.append(c)
    return t

###
def squarePerimeter(n):
    result = 0
    for i in range(4):
        result += n
    return result

###
def robotWalk(a):
    minX = 0
    minY = -1
    maxX = float('inf')
    maxY = float('inf')

    x = 0
    y = 0

    for i in range(len(a)):
        j = i % 4

        if j == 0:
            y += a[i]
            if y >= maxY:
                return True
            maxY = y

        elif j == 1:
            x += a[i]
            if x >= maxX:
                return True
            maxX = x

        elif j == 2:
            y -= a[i]
            if y <= minY:
                return True
            minY = y

        elif j == 3:
            x -= a[i]
            if (x <= minX): return True
            minX = x

    return False

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

###
def maximizeNumberRoundness(n):
    tmp = n
    zeros = 0
    while tmp:
        if tmp % 10 == 0:
            zeros += 1
        tmp /= 10
    result = zeros
    for i in range(zeros):
        if n % 10 == 0:
            result -= 1
        n /= 10
    return result

###
def sameDigitNumber(n):
    digit =  n % 10
    while n != 0:
        if n % 10 != digit:
            return False
        n /= 10
    return True

###
def properNounCorrection(noun):
    return noun.title()

###
def coolString(inputString):

    def isLowercase(symbol):
        if 'a' <= symbol <= 'z':
            return True
        return False

    def isUppercase(symbol):
        if 'A' <= symbol <= 'Z':
            return True
        return False

    firstIsLowercase = isLowercase(inputString[0])
    firstIsUppercase = isUppercase(inputString[0])

    if not (firstIsLowercase or firstIsUppercase):
        return False

    for i in range(1, len(inputString)):
        if i % 2 != 0:
            if (isLowercase(inputString[i]) == firstIsLowercase or
                    isUppercase(inputString[i]) == firstIsUppercase):
                return False
        else:
            if (isLowercase(inputString[i]) != firstIsLowercase or
                    isUppercase(inputString[i]) != firstIsUppercase):
                return False

    return True

###

def squareDigitsSequence(a0):

    cur = a0
    was = set()

    while not (cur in was):
        was.add(cur)
        nxt = 0
        while cur > 0:
            nxt += (cur % 10) * (cur % 10)
            cur /= 10
        cur = nxt
        print cur

    return len(was) + 1

###
def powersOfTwo(n):
    r = []
    t = 1
    while (n > 0):
        if (n % 2 == 1):
            r.append(t)
        n >>= 1
        t <<= 1

    return r

###
def liquidMixing(densities):
    result = [densities[0]]
    for i in range(1, len(densities)):
        for j in range(i + 1):
            if densities[i] <= densities[j]:
                tmp = densities[i]
                for k in range(i, j, -1):
                    densities[k] = densities[k - 1]
                densities[j] = tmp
                if i % 2 == 1:
                    result.append((densities[(i + 1) / 2] +
                                  densities[i / 2]) / 2.0)
                else:
                    result.append(densities[i / 2])
                break
    return result

###
def numberOfSolutions(n):

    result = 0
    for a in range(n + 1, 2 * n):
        if (a * n) % (a - n) == 0:
            result += 1

    return result * 2 + 1

###
def areSimilarNumbers(a, b, divisor):
    if (a % divisor == 0 and b % divisor == 0
      or a % divisor != 0 and b % divisor != 0):
        return True
    return  False

###
def maximalAllowableSubarrays(inputArray, maxSum):
    for i in range(len(inputArray)):
        t = inputArray[i]
        for j in range(i+1, len(inputArray)):
            t += inputArray[j]
            if (t > maxSum):
                inputArray[i] = j - 1
                break

        if (t <= maxSum):
            inputArray[i] = len(inputArray) - 1

    return inputArray

###
def patternMatching(inputStr, pattern):

    dp = []
    for i in range(len(inputStr) + 1):
        line = []
        for j in range(len(pattern) + 1):
            line.append(False)
        dp.append(line)

    dp[0][0] = True
    for i in range(len(inputStr) + 1):
        for j in range(len(pattern)):
            if not dp[i][j]:
                continue
            if (i < len(inputStr)
            and (inputStr[i] == pattern[j] or pattern[j] == '?')):
                dp[i + 1][j + 1] = True
            if pattern[j] == '*':
                for k in range(len(inputStr) - i + 1):
                    dp[i + k][j + 1] = True

    return dp[len(inputStr)][len(pattern)]

###
def firstReverseTry(arr):
    if not arr:
        return arr
    arr[0], arr[-1] = arr[-1], arr[0]
    return arr

###
def mixedFractionToImproper(a):

    b = [a[1] + a[0] * a[2], a[2]]
    return b

###
def differentDigitsNumberSearch(inputArray):

    for i in range(len(inputArray)):
        cur =  inputArray[i]
        was = [False] * 10
        ok = True
        while cur > 0:
            if was[cur % 10]:
                ok = False
                break
            was[cur % 10] = True
            cur /= 10
        if ok:
            return inputArray[i]

    return -1

###
def telephoneGame(messages):
    answer = -1
    for i in range(1, len(messages)):
        if messages[i] != messages[i - 1]:
            answer = i
            break
    return answer

###
def repetitionEncryption(letter):
    pattern = r"([a-zA-Z]+)[^a-zA-Z]+\1(?![a-zA-Z])"
    regex = re.compile(pattern,re.IGNORECASE)
    return len(re.findall(regex, letter))

###
def squareDigitsSequence(a0):

    def helper(a0, l):
        if a0 in l:
            return len(l) + 1
        l.append(a0)
        n = [int(x) for x in list(str(a0))]
        s = 0
        for i in n:
            s+= (i**2)

        return helper(s, l)

    return helper(a0, [])

###
def maxDigit(n):

    result = 0
    while n > 0:
        result = max(result, n % 10)
        n /= 10

    return result

###
def divisorsPairs(sequence):

    result = 0

    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if  sequence[i] % sequence[j] == 0 or sequence[j] % sequence[i] == 0 :
                result += 1

    return result

###
def easyAssignmentProblem(skills):
    if skills[0][0] - skills[1][0] > skills[0][1] - skills[1][1]:
        return [1, 2]

    return [2, 1]

###
def knapsackLight(value1, weight1, value2, weight2, maxW):

    if weight1 + weight2 <= maxW:
        return value1 + value2
    if min(weight1, weight2) > maxW:
        return 0
    if weight1 <= maxW and (value1 >= value2 or weight2 > maxW):
        return value1
    return value2

###
def isIPv4Address(inputString):

    currentNumber = 0
    emptyField = True
    countNumbers = 0

    inputString += '.'

    for i in range(len(inputString)):
        if inputString[i] == '.':
            if emptyField:
                return False
            countNumbers += 1
            currentNumber = 0
            emptyField = True
        else:
            digit = ord(inputString[i]) - ord('0')
            if digit < 0 or digit > 9:
                return False
            emptyField = False
            currentNumber = currentNumber * 10 + digit
            if currentNumber > 255:
                return False
    return countNumbers == 4

###
def segmentSumsMatrix1(inputArray):

    answer = []
    for i in range(len(inputArray)):
        line = []
        for j in range(len(inputArray)):
            line.append(0)
        answer.append(line)

    for i in range(len(inputArray)):
        for j in range(i, -1, -1):
            for k in range(i, len(inputArray)):
                answer[j][k] += inputArray[i]
                answer[k][j] += inputArray[i]
        answer[i][i] -= inputArray[i]

    return answer

###
def divisorsSuperset(superset, n):

    def isInSequence(sequence, elem):
        for i in range(len(sequence)):
            if (sequence[i] == elem):
                return True
        return False

    res = 0

    for i in range(1, n + 1):
        correct = True
        j = 2
        while j * j <= i:
            if i % j == 0 :
                if not isInSequence(superset, j) or not isInSequence(superset, i / j):
                    correct = False
                    break
            j += 1
        if correct:
            res += 1

    return res

###
def addBorder(picture):
    l = len(picture[0])
    picture = ["*"*l] + picture + ["*"*l]
    picture = [("*" + x + "*") for x in picture]
    return picture

###
def sortByLength(inputArray):

    for i in range(len(inputArray)):
        minIndex = i
        tmp = inputArray[i]
        for j in range(i + 1, len(inputArray)):
            if len(inputArray[j]) < len(inputArray[minIndex]):
                minIndex = j
        inputArray[i] = inputArray[minIndex]
        inputArray[minIndex] = tmp

    return inputArray

###
def twoArraysNthElement(array1, array2, n):

    def lowerBound(array, elem):
        l = -1
        r = len(array)
        while l + 1 < r :
            mid = (l + r) / 2
            if array[mid] <= elem:
                l = mid
            else:
                r = mid
        return l

    l = -1
    r = len(array1)

    while l + 1 < r:
        mid = (l + r) / 2
        pos = lowerBound(array2, array1[mid])

        if mid + pos + 1 <= n:
            l = mid
        else:
            r = mid

    if l > -1 and l + lowerBound(array2, array1[l]) + 1 == n:
        return array1[l]
    return twoArraysNthElement(array2, array1, n)

###
def perfectTeamOfMinimalSize(n, candidates):

    MAX_MASK = 1 << n

    knowledge = [0] * len(candidates)
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            knowledge[i] = knowledge[i] | (1 << candidates[i][j])
    teamSize = [-1] * MAX_MASK
    teamSize[0] = 1
    for i in range(len(teamSize)):
        if teamSize[i] == -1:
            continue
        for j in range(len(knowledge)):
            i2 = i ^ knowledge[j]
            if teamSize[i2] == -1 or teamSize[i2] > teamSize[i] + 1:
                teamSize[i2] = teamSize[i] + 1

    return teamSize[MAX_MASK - 1]

###
def fileNaming(names):
    def calculateHash(inputString):
        P = 997
        M = 28001
        hashValue = 0
        for i in range(len(inputString)):
            hashValue = (hashValue * P + ord(inputString[i])) % M
        return hashValue

    hashMapSize = len(names) * 2
    ##
    #     Information about the string in the hash map
    #     is stored in the following way:
    #     [string itself,
    #      its hash,
    #      the smallest possible integer to use with this name]
    ##
    hashMap = []
    result = []

    def searchHM(position, hashValue):
        while (hashMap[position][0] != ''
          and hashMap[position][1] != hashValue):
            position = (position + 1) % hashMapSize
        return position

    for i in range(hashMapSize):
        hashMap.append(['', -1, 0])

    for i in range(len(names)):
        hashValue = calculateHash(names[i])
        startPos = searchHM(hashValue % hashMapSize, hashValue)
        if hashMap[startPos][0] == '':
            hashMap[startPos] = [names[i], hashValue, 1]
            result.append(names[i])
        else:
            newName = names[i] + '(' + str(hashMap[startPos][2]) + ')'
            newNameHash = calculateHash(newName)
            position = searchHM(newNameHash % hashMapSize, newNameHash)

            while hashMap[position][0] != '':
                hashMap[startPos][2] += 1
                newName = names[i] + '(' + str(hashMap[startPos][2]) + ')'
                newNameHash = calculateHash(newName)
                position = searchHM(newNameHash % hashMapSize, newNameHash)
            hashMap[position] = [newName, newNameHash, 1]
            result.append(newName)
            hashMap[startPos][2] += 1

    return result

###
def prefixSumsToSuffixSums(prefixSums):
    def helper(prefixSums):
        prefixSums = list(reversed(prefixSums))
        return list(reversed([a - b for (a, b) in zip(prefixSums, prefixSums[1:])] + [prefixSums[-1]]))

    prefixSums = helper(prefixSums)
    t = []
    for i in range(len(prefixSums)):
        t.append(sum(prefixSums[i:]))
    return t

###
def prefixSumsToSuffixSums(prefixSums):
    t = []
    t.append(prefixSums[len(prefixSums) - 1])

    for i in range(1, len(prefixSums)):
        t.append(prefixSums[len(prefixSums) - 1] - prefixSums[i - 1])

    return t

###
def myMin(a, b):
    if a < b:
        return a
    return b

###
def checkSameElementExistence(arr1, arr2):
    a = set(arr1)
    b = set(arr2)
    return True if len(a & b) > 0 else False

###
def isSkewSymmetricMatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if  matrix[i][j] != -matrix[j][i] :
                return False
    return True

###
def summaryProficiency(a, n, m):
    t = i = 0
    while n > 0:
        if (a[i] >= m):
            t += a[i]
            n-= 1
        i+= 1
    return t

###
def primeFactors(n):
    factors = []
    divisor = 2

    while n >= 2:
        if n % divisor == 0:
            factors.append(divisor)
            n /= divisor
        else:
            divisor += 1
    return factors

###
def binarySearch(inputArray, searchElement):

    minIndex = -1
    maxIndex =  len(inputArray) - 1

    while minIndex < maxIndex - 1:
        currentIndex = (minIndex + maxIndex) / 2
        currentElement = inputArray[currentIndex]

        if currentElement < searchElement:
            minIndex = currentIndex
        else:
            maxIndex = currentIndex

    if maxIndex == len(inputArray) or inputArray[maxIndex] != searchElement:
        return -1
    return maxIndex

###
def passwordCheckRegExp(inputString):
    t1 = len(re.findall(r'([A-Z])', inputString))
    t2 = len(re.findall(r'([a-z])', inputString))
    t3 = len(re.findall(r'([0-9])', inputString))

    if (len(inputString) >= 5) and t1 != 0 and t2 != 0 and t3 != 0:
        return True

    return False
###
def maxSubmatrixSum(matrix, n, m):

    result = 0
    for i in range(len(matrix) - n + 1):
        for j in range(len(matrix[0]) - m + 1):
            sumValue = 0
            for x in range(n):
                for y in range(m):
                    sumValue += matrix[i + x][j + y]
            if i == 0 and j == 0 or sumValue > result:
                result = sumValue

    return result

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
    return max(mx[0] - mn[0], mx[1] - mn[1])

###
def checkSameElementExistence(arr1, arr2):
    x = set(arr1)
    y = set(arr2)

    return len(x & y) > 0

###
def evenDigitsOnly(n):

    if n == 0:
        return True
    if n % 2 != 0:
        return False
    return evenDigitsOnly(n / 10)

###
def isMonotonous(sequence):
    if len(sequence) == 1:
        return True
    direction =  sequence[1] - sequence[0]
    for i in range(0, len(sequence) - 1):
        if direction * (sequence[i + 1] - sequence[i]) <= 0:
            return False
    return True

###
def arrayConversion(l):
    i = 1
    A = []
    B = []
    while len(l) > 1:
        if i == 1:
            i = 2
            A = l[::2]
            B = l[1::][::2]
            l = [a+b for a,b in zip(A,B)]
        else:
            i = 1
            A = l[::2]
            B = l[1::][::2]
            l = [a*b for a,b in zip(A,B)]

    return l[0]

###
def appleBoxes(k):
    sum = 0
    x = 0
    while x <= k:
        if x % 2 == 0:
            sum += x * x
        else:
            sum -= x * x
        x += 1

    return sum

###
def howManySundays(n, startDay):
    week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday']
    startIndex = -1

    for i in range(len(week)):
        if week[i] == startDay:
            startIndex = i
            break

    return  (n + startIndex) / 7

###
def maxDigit(n):
    return int(sorted(list(str(n)))[-1])

###
def countSumOfTwoRepresentations(n, l, r):
    result = 0

    for a in range(l, r + 1):
        b = a
        while b <= r:
            if a + b == n:
                result += 1
            b += 1

    return result

###
def fractionSubtraction(a, b):

    def gcdEuclid(a, b):
        if a == 0:
            return b
        return  gcdEuclid(b % a, a)

    c = [a[0] * b[1] - a[1] * b[0], a[1] * b[1]]
    gcd = gcdEuclid(c[0], c[1])

    c[0] /= gcd
    c[1] /= gcd

    return c

###
def telephoneGame(m):
    t = -1
    for i in range(len(m) - 1):
        if m[i] != m[i + 1]:
            t = i + 1
            break
    return t

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
def truncateString(s):

    def truncate(l, r):
        if l >= r:
            return ''
        newL = l
        newR = r
        left = ord(s[l]) - ord('0')
        right = ord(s[r - 1]) - ord('0')
        if left % 3 == 0:
            newL+= 1
        elif right % 3 == 0:
            newR -= 1
        elif (left + right) % 3 == 0:
            newL += 1
            newR -= 1
        else:
            return s[l : r]

        return truncate(newL, newR)

    return truncate(0, len(s))

###
def sumOfMultiples(n, k):
    sum = 0
    for i in range(k, n+1):
        if (i % k == 0): sum += i
    return sum

###
def insideCircle(point, center, radius):

    def sqr(value):
        return value * value

    if sqr(point[0] - center[0]) + sqr(point[1] - center[1]) <= sqr(radius):
        return True
    return False

###
def factorSum(n):
    def primeFacts(n):
        list= []
        i = 2
        while i<=n:
            while n%i == 0:
                list.append(i)
                n /= i
            i+= 1
        return list

    while sum(primeFacts(n)) != n:
        n = sum(primeFacts(n))
    return n

###
def isPower(n):
    for i in range(401):
        for j in range(2,401):
            if i**j == n:
                return True
            elif i**j > n:
                break
    return False

###
def generatePalindromes(charactersSet):

    result = []

    N = len(charactersSet)
    palindrome = [0] * N
    letterCnt = [0] * 26

    for i in range(N):
        letterCnt[ord(charactersSet[i]) - ord('a')] += 1
    if N % 2 == 1:
        for i in range(26):
            if letterCnt[i] % 2 == 1:
                letterCnt[i] -= 1
                palindrome[N / 2] = chr(ord('a') + i)
                break

    def generate(idx):
        if idx >= (N) / 2:
            result.append(''.join(palindrome))
            return
        for i in range(26):
            if letterCnt[i] >= 2:
                letterCnt[i] -= 2
                palindrome[idx] = chr(ord('a') + i)
                palindrome[N - idx - 1] = chr(ord('a') + i)
                generate(idx + 1)
                letterCnt[i] += 2

    generate(0)
    return result

###
def truncateString(s):

    def truncate(l, r):
        if l >= r:
            return ''
        newL = l
        newR = r
        left = ord(s[l]) - ord('0')
        right = ord(s[r - 1]) - ord('0')
        if left % 3 == 0:
            newL += 1
        elif right % 3 == 0:
            newR -= 1
        elif (left + right) % 3 == 0:
            newL += 1
            newR -= 1
        else:
            return s[l : r]

        return truncate(newL, newR)

    return truncate(0, len(s))

###
def equidistantTriples(coordinates):

    ans = 0
    for i in range(1, len(coordinates)):
        left = i - 1
        right = i + 1
        while left >= 0 and right < len(coordinates):
            distL = - coordinates[left] + coordinates[i]
            distR = coordinates[right] - coordinates[i]
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
def constructSubmatrix(matrix, rowsToDelete, columnsToDelete):

    res = []
    useRow = []
    useColumn = []

    for i in range(len(matrix)):
        useRow.append(True)
    for i in range(len(matrix[0])):
        useColumn.append(True)

    for i in range(len(rowsToDelete)):
        useRow[ rowsToDelete[i] ] = False
    for i in range(len(columnsToDelete)):
        useColumn[ columnsToDelete[i] ] = False

    for i in range(len(matrix)):
        if useRow[i]:
            res.append([])
            for j in range(len(matrix[0])):
                if useColumn[j]:
                    res[len(res) - 1].append(matrix[i][j])

    return res

###
def rectanglesIntersection(a, b, c, d):
    intersection = []
    for i in range(2):
        if a[i] > b[i]:
            t = a[i]
            a[i] = b[i]
            b[i] = t
        if c[i] > d[i]:
            t = c[i]
            c[i] = d[i]
            d[i] = t
        if  b[i] < c[i] or a[i] > d[i]:
            return 0
        intersection += [min(b[i], d[i]) - max(a[i], c[i])]

    return intersection[0] * intersection[1]

###
def swapCase(text):
    return text.swapcase()

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

###
def charactersRearrangement(string1, string2):
    characters1 = list(string1)
    characters2 = list(string2)
    correct = True

    characters1.sort()
    characters2.sort()

    for i in range(max(len(characters1), len(characters2))):
        if (i >= len(characters1) or i >= len(characters2)
           or characters1[i] != characters2[i]):
            correct = False
            break

    return correct

###
def arrayKthGreatest(inputArray, k):

    for i in range(k):
        indexOfMaximum = i
        tmp = inputArray[i]

        for j in range(i + 1, len(inputArray)):
            if inputArray[j] > inputArray[indexOfMaximum]:
                indexOfMaximum = j

        inputArray[i] = inputArray[indexOfMaximum]
        inputArray[indexOfMaximum] = tmp

    return  inputArray[k - 1]

###
def houseNumbersSum(inputArray):
    s = 0
    for a in inputArray:
        s += a
        if a == 0:
            return s

###
def arrayComplexElementsProduct(real, imag):
    answer = [real[0], imag[0]]
    for i in range(1, len(real)):
        tmp = answer[0] * real[i] - answer[1] * imag[i]
        answer[1] =  answer[0] * imag[i]+answer[1]*real[i]
        answer[0] = tmp
    return answer

###
def arrayCenter(a):
    avg = float(sum(a))/len(a)
    print avg
    m = min(a)
    return [x for x in a if abs(x - avg) < m]

###
def countSumOfTwoRepresentations2(n, l, r):
    a = max(min(n - 2*l, 2*r - n), -2)
    return 1+a/2

###
def isUppercase(symbol):
    if ('A' <= symbol <= 'Z'):
        return True
    return False

###
def factorial(n):
    if n == 0:
        return 1

    return  n * factorial(n-1)

###
def regularBracketSequence2(s):
    stack, lookup = [], {"(": ")", "{": "}", "[": "]"}
    for each in s:
            if each in lookup:
                stack.append(each)
            elif len(stack) == 0 or lookup[stack.pop()] != each:
                return False
    return len(stack) == 0

###
def regularBracketSequence2(sequence):
    res=0
    last=sequence[0]
    for i in sequence:
        if i=='(':
            res+=1
        if i==')':
            res-=1
        if i=='[':
            res+=2
        if i==']':
            res-=2
        if res<0 or last=='(' and i==']' or last=='[' and i==')':
            return 0
        last=i
    return not res

###
def isPrime(n):
    if (n == 2): return True
    if (n % 2 == 0): return False
    
    i = 3
    while i*i <=n:
        if n % i == 0:
            return False
        i+= 2
    
    return True
