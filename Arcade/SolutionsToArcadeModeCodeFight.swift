//
//  SolutionsToArcadeModeCodeFight.swift
//  tryTest
//
//  Created by Himanshi Bhardwaj on 7/6/17.
//  Copyright © 2017 HPP. All rights reserved.
//

import Foundation

/*
 INTRO
 */
func add(param1: Int, param2: Int) -> Int {
    return param1 + param2
}

func centuryFromYear(year: Int) -> Int {
    return year % 100 != 0 ? year / 100 + 1 : year / 100
}

func checkPalindrome(inputString: String) -> Bool {
    return inputString == String(inputString.characters.reversed())
}

/*
 THE CORE
 */
func addTwoDigits(n: Int) -> Int {
    return n%10 + n/10
}

func largestNumber(n: Int) -> Int {
    var output = ""
    for i in (1...n) {
        output += String(9)
        print(i, n, output)
        
    }
    
    return Int(output)!
}


func candies(n: Int, m: Int) -> Int {
    return (m / n) * n
}


func seatsInTheater(nCols: Int, nRows: Int, col: Int, row: Int) -> Int {
    return (nCols - col + 1) * (nRows - row)
}

func maxMultiple(divisor: Int, bound: Int) -> Int {
    var output = 1
    while true {
        if (output % divisor == 0) && (output > bound) {
            break
        }
        output += 1
    }
    
    return output - divisor
}

func circleOfNumbers(n: Int, firstNumber: Int) -> Int {
    return (firstNumber + n/2) % n
}

func lateRide(n: Int) -> Int {
    let hour = n/60
    let min = n%60
    return hour/10 + hour%10 + min/10 + min%10
}

func phoneCall(min1: Int, min2_10: Int, min11: Int, s: Int) -> Int {
    var minutes = 0
    var centsLeft = s
    if (centsLeft < min1) {
        return 0
    }
    
    if (centsLeft == min1) {
        return 1
    }
    
    centsLeft = centsLeft - min1
    minutes = 1
    //print(minutes, centsLeft)
    if (min2_10*9 >= centsLeft && centsLeft >= min1) {
        print("here")
        minutes += centsLeft/min2_10
        return minutes
    }
    
    centsLeft = centsLeft - min2_10*9
    minutes += 9
    //print(minutes, centsLeft)
    minutes += centsLeft/min11
    return minutes
}

func reachNextLevel(experience: Int, threshold: Int, reward: Int) -> Bool {
    return threshold <= experience + reward
    
}

func knapsackLight(value1: Int, weight1: Int, value2: Int, weight2: Int, maxW: Int) -> Int {
    if (weight1 + weight2) <= maxW {
        return value1 + value2
    }
    
    if min(weight1, weight2) > maxW {
        return 0
    }
    
    if weight1 <= maxW && (value1 >= value2 || weight2 > maxW) {
        return value1
    }
    
    return value2
    
}

func extraNumber(a: Int, b: Int, c: Int) -> Int {
    var output = (a == b) ? c : (a == c ? b: a )
    
    return output
}

func isInfiniteProcess(a: Int, b: Int) -> Bool {
    var a = a
    var b = b
    while (a != b) {
        a += 1;
        b -= 1;
        if (a == b) {
            return false;
        }
        if (a > b) {
            return true;
        }
    }
    return false;
}

func arithmeticExpression(a: Int, b: Int, c: Int) -> Bool {
    return (a + b == c) || (a - b == c) || (a * b == c) || ((a / b == c) && (a % b == 0))
    
}

func tennisSet(score1: Int, score2: Int) -> Bool {
    var score1 = score1
    var score2 = score2
    if score1 < score2 {
        var t = score1
        score1 = score2
        score2 = t
    }
    
    print(score1, score2)
    if (score1 == 6 && score2 <= 4) {
        return true
    }
    
    if (score1 == 7 && 5 <= score2 && score2 < 7) {
        return true
    }
    
    return false
}

func willYou(young: Bool, beautiful: Bool, loved: Bool) -> Bool {
    return (young && beautiful && !loved) || (loved && (!young || !beautiful))
}

func metroCard(lastNumberOfDays: Int) -> [Int] {
    return lastNumberOfDays == 28 || lastNumberOfDays == 30 ? [31] : [28, 30, 31]
}

func killKthBit(n: Int, k: Int) -> Int {
    return n & ~Int(pow(2.0, Double(k-1)))
    
}

func arrayPacking(a: [Int]) -> Int {
    var t = ""
    var bin = ""
    var zerosToPad = 0
    
    for each in a {
        bin = String(each, radix: 2)
        zerosToPad = (8 - bin.characters.count)
        while zerosToPad > 0 {
            bin = "0" + bin
            zerosToPad -= 1
        }
        t = bin + t
    }
    return Int(t, radix: 2)!
    
}

func rangeBitCount(a: Int, b: Int) -> Int {
    var t = ""
    for each in a...b {
        t += String(each, radix:2)
    }
    
    var components = t.components(separatedBy: "1")
    
    return components.count - 1
    
}

func mirrorBits(a: Int) -> Int {
    var bin = String(a, radix: 2)
    bin = String(bin.characters.reversed())
    return Int(bin, radix: 2)!
    
}

func secondRightmostZeroBit(n: Int) -> Int {
    return (n | (n+1) + 1) & ~(n | (n+1))
}

func swapAdjacentBits(n: Int) -> Int {
    return (n>>1) & 0x55555555 | (n<<1) & 0xAAAAAAAA
}

func differentRightmostBit(n: Int, m : Int) -> Int {
    return  (n % 2 != m % 2) ? 1 : 2 * differentRightmostBit(n: n / 2, m: m / 2)
}

func equalPairOfBits(n: Int, m : Int) -> Int {
    return (n & 1) != (m & 1) ? equalPairOfBits(n: n>>1, m: m>>1)*2 : 1
}

func leastFactorial(n: Int) -> Int {
    var k = 1, m = 1
    while k < n {
        k *= m
        m += 1
    }
    return k
}

