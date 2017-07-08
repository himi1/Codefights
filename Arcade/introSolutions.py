def add(param1, param2):
    return param1 + param2

#
def centuryFromYear(year):
    return year/100 + 1 if year%100 != 0 else year /100

#
def checkPalindrome(inputString):
    return inputString == inputString[::-1]

#
def adjacentElementsProduct(inputArray):
    output = - sys.maxint - 1
    for i in range(1, len(inputArray)):
        if inputArray[i]*inputArray[i-1] > output:
            output = inputArray[i]*inputArray[i-1]

    return output

#
def shapeArea(n):
    area = 1
    for i in range(n):
        area += 4 * i

    return area

#
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

#
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

#
def matrixElementsSum(matrix):
    output = 0
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            if matrix[j][i] == 0:
                break
            output += matrix[j][i]
    return output

#
def allLongestStrings(inputArray):
    longestLength = max([len(x) for x in inputArray])
    output = []
    for each in inputArray:
        if len(each) == longestLength:
            output.append(each)

    return output
