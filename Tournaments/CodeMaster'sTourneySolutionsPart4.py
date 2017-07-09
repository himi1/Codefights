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
