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
