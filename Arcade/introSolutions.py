def add(param1, param2):
    return param1 + param2

###
def centuryFromYear(year):
    return year/100 + 1 if year%100 != 0 else year /100

###
def checkPalindrome(inputString):
    return inputString == inputString[::-1]

###
def adjacentElementsProduct(inputArray):
    output = - sys.maxint - 1
    for i in range(1, len(inputArray)):
        if inputArray[i]*inputArray[i-1] > output:
            output = inputArray[i]*inputArray[i-1]

    return output

###
def shapeArea(n):
    area = 1
    for i in range(n):
        area += 4 * i

    return area

###
def makeArrayConsecutive2(statues):
    statues = sorted(statues)
    end = statues[-1]
    i = statues[0]
    count = 0
    while i < end:
        i+= 1
        if i not in statues:
            count+= 1

    return count

###
def almostIncreasingSequence(s):
    i = 0
    j = 1
    count = 0
    while i < len(s)-1 and j < len(s):
        if s[i] >= s[j]:
            if count == 1:
                return False
            count += 1
            if i != 0:
                if s[i-1] >= s[j]:
                    j += 1
                    continue
        i = j
        j += 1

    return True

###
def matrixElementsSum(matrix):
    output = 0
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            if matrix[j][i] == 0:
                break
            output += matrix[j][i]
    return output

###
def allLongestStrings(inputArray):
    longestLength = max([len(x) for x in inputArray])
    output = []
    for each in inputArray:
        if len(each) == longestLength:
            output.append(each)

    return output

###
def isPowerOfTwo(n):

    while n % 2 == 0:
        n >>= 1

    if n == 1:
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
            digitSum +=  int(mask[i]) - leftBound
        else:
            asteriskPos = i

    for i in range(10):
        if (digitSum + i) % 3 == 0:
            mask[asteriskPos] = chr(leftBound + i)
            if (ord(mask[len(mask) - 1]) - leftBound) % 2 == 0:
                answer.append(''.join(mask))

    return answer

###
def chessBoardSquaresUnderQueenAttack(a, b):
    t = 0
    for i in range(a):
        for j in range(b):
            for dx in range(a):
                for dy in range(b):
                    if (i == dx or j == dy) or abs(i-dx) == abs(j-dy):
                        t+= 1

    return t - a*b

###
def regularBracketSequence2(sequence):

    stack = []
    for i in range(len(sequence)):
        if (len(stack) > 0
            and stack[len(stack) - 1] == '(' and sequence[i] == ')'):
            stack.pop()
            continue
        if (len(stack) > 0
            and stack[len(stack) - 1] == '[' and sequence[i] == ']'):
            stack.pop()
            continue
        stack.append(sequence[i])

    if len(stack) != 0:
        return False
    return True
