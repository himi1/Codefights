'''
Given an array of words and a length l, format the text such that each line has
exactly l characters and is fully justified on both the left and the right.
Words should be packed in a greedy approach; that is, pack as many words as
possible in each line. Add extra spaces when necessary so that each line has
exactly l characters.

Extra spaces between words should be distributed as evenly as possible. If the
number of spaces on a line does not divide evenly between words, the empty slots
 on the left will be assigned more spaces than the slots on the right. For the
 last line of text and lines with one word only, the words should be left
 justified with no extra space inserted between them.

Example:
For
words = ["This", "is", "an", "example", "of", "text", "justification."]
and l = 16, the output should be

textJustification(words, l) = ["This    is    an",
                               "example  of text",
                               "justification.  "]
'''

def textJustification(words, l):
    temp = 0
    result = []

    def justifyLine(line, lastLine = False):
            # add justification to line
            # print ("here i m:")
            t = l - len(line)
            # print t, len(line)
            if wordInLine == 1 or lastLine:
                line += " "*t
                return line

            line = line.replace(" ", " " + "#"*(t/(wordInLine - 1)))
            # print ("t: ", t, "wordInLine: ", wordInLine)
            t = t%(wordInLine - 1)
            # print line
            # print ("%t: ", t)
            while t:
                # print line
                line = line.replace(" ", "##", 1)
                t -= 1
                # print line
            line = line.replace("#", " ")
            return line

    #to be reset every time a line is written to result
    wordInLine = 0
    line = ""
    for word in words:
        if line == "":
            line += word
            wordInLine+= 1
        elif len(line) + len(word) + 1 <= l: # +1 is for " "
            line+= " " + word
            wordInLine+= 1
        else:
            line = justifyLine(line)
            # add to result
            result.append(line)
            #reset line
            line = ""
            wordInLine = 0

            line += word
            wordInLine+= 1


    if line != "":
        wordInLine = len(line.split())
        line = justifyLine(line, True)
        # add to result
        result.append(line)
    else:
        result.append(" "*l)

        #print ("word: ", word, " | line: ", line)

    return result
