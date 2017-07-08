def countIncreasingSequences(n, k):

    ##
    #  list dp (short for dynamic programming)
    #  is used for storing the interim values.
    ##
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

#
def maxDivisor(left, right, divisor):

    i = right
    while i >= left:
        if i % divisor == 0:
            return i
        i -= 1
    return  -1

#
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

#
def candles(candlesNumber, makeNew):
    burned = 0
    leftovers = 0
    while candlesNumber > 0:
        burned += candlesNumber
        leftovers += candlesNumber
        candlesNumber = leftovers / makeNew
        leftovers %= makeNew
    return burned

#
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

#
def isDigit(symbol):
    return symbol.isdigit()

#
def crossingSum(matrix, row, column):

    result = 0
    for i in range(len(matrix)):
        result += matrix[i][column]
    for i in range(len(matrix[0])):
        result += matrix[row][i]
    result -= matrix[row][column]

    return result

#
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

#
def isInfiniteProcess(a, b):
    while (a != b):
        a+= 1
        b-= 1
        if (a == b): return False
        if (a > b): return True

    return False

#
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

#
def gcdNaive(a, b):

    gcd = 1
    for divisor in range(2, min(a, b) + 1):
        if a % divisor == 0 and b % divisor == 0:
            gcd =  divisor

    return gcd

#
def permutationShift(permutation):
    min = max = 0
    for i in range(len(permutation)):
        if (permutation[i] - i > max):
            max = permutation[i] - i
        if (permutation[i] - i < min):
            min = permutation[i] - i

    return max - min

#
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

#
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

#
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

#
def arrayMinimumIndex(inputArray):

    indexOfMinimum = 0
    for i in range(1, len(inputArray)):
        if inputArray[i] < inputArray[indexOfMinimum]:
            indexOfMinimum = i
    return indexOfMinimum

#
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

#
def halvingSum(n):
    sum = 0
    i = n
    while i > 0:
        sum += i
        i /= 2
    return sum

#
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

#
def hailstoneSequence(n):
    t = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        t += 1
    return t

#
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

#
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

#
def maxSubarray(inputArray):
        currentMax = 0
        result = 0

        for i in range(len(inputArray)):
                currentMax = max(0, currentMax + inputArray[i])
                result = max(result, currentMax)

        return result

#
def piecesOfDistinctLengths(strawLength):
    i = 1
    while ((i + 1) * (i + 2) / 2 <= strawLength):
        i += 1

    return i

#
def piecesOfDistinctLengths(strawLength):
    t = 1
    s = 0
    while s + t <= strawLength:
        s += t
        t += 1
    return t - 1

#
def checkIncreasingSequence(seq):

    for i in range(1, len(seq)):
        if seq[i] <= seq[i + 1]:
            return False

    return True

#
def bettingGame(l):

    s = 0
    for i in range(len(l)):
        s += l[i]
    if s == 0:
        return False

    return  s % len(l) == 0

#
def isInformationConsistent(evidences):
    for i in range(len(evidences[0])):
        t = 0
        for j in range(len(evidences)):
            if evidences[j][i] * t < 0:
                return False
            else:
                t += evidences[j][i]
    return True

#
def applesDistribution(apples, boxCapacity, maxResidue):
    result = 0
    for i in range(1, boxCapacity + 1):
        if apples % boxCapacity <= maxResidue:
            result += 1
    return result

#
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

#
def isSuspiciousRespondent(ans1, ans2, ans3):
     return  (ans1 and ans2 and ans3) or (not ans1 and not ans2 and not ans3)

#
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

#
def arePrizesOK(first, second, third):
    if first < second:
        return  False
    if second < third:
        return False
    return True

#
def howManySundays(n, startDay):
    dict = {"Monday" : 1, "Tuesday" : 2, "Wednesday":3, "Thursday" : 4, "Friday" : 5,"Saturday":6, "Sunday":0}

    t = n + dict[startDay]
    return t/7

#
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

#
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

#
def charactersRearrangement(string1, string2):
    string1 = sorted(string1)
    string2 = sorted(string2)
    return string1==string2

#
def isCorrectSentence(inputString):

    leadChar = inputString[0]
    endChar = inputString[len(inputString) - 1]

    if ('A' <= leadChar and leadChar <= 'Z'
          and endChar == '.'):
        return True
    else:
        return False

#
def digitCharactersSum(ch1, ch2):
    x1 = ord(ch1) - ord('0')
    x2 = ord(ch2) - ord('0')
    if x1 + x2 < 10:
        return chr(ord('0') + x1 + x2)
    else:
        return '1' + chr(ord('0') + (x1 + x2) % 10)

#
def isSmooth(arr):
    l = len(arr)
    if l%2 == 0: t = arr[l/2]+arr[l/2-1]
    else: t = arr[l//2]

    if arr[0]==t and arr[-1]==t:
        return True
    return False

#
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

#
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

#
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

#
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

#
def arithmeticProgression(element1, element2, n):
    return 0 if (element2 < element1) else (element1 + (n - 1) * abs(element1 - element2))
