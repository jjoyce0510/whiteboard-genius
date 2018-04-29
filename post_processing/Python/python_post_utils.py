# author: Robert Simari
# date  : 4/25/18
# functions designed for doing post processing for Python programs


class Python_Post_Processor(object):
    def __init__(self):
        # Python Keywords
        self.keywords = ["False", "class", "finally", "is", "return", "None", "continue", "for", "lambda", "try", 
        "True", "def", "from", "while", "and", "del", "global", "not", "with", "as", "elif", "if", "or", "yield", 
        "assert", "else", " import", "pass", "break", "except", "in", "raise"]

        self.operators = ['+', '-', '%', '*', '/', '//']

        self.assignment = ['=']

        self.delimiters = [' ','+','-','*','/',',',';','>','<','=','(',')','[',']','{','}']

        self.brackets = ['(',')','[',']','{','}']

        self.line_terms = [':']

        # should be a tree of tokens with the children being the next possible token type
        self.line_orders = []
        # for loop tokens
        self.line_orders.append(['keyword', 'variable', 'keyword', 'variable', 'end of line'])
        # declaration tokens
        self.line_orders.append(['variable', 'assignment', 'variable'])
        self.line_orders.append(['variable', 'assignment', 'literal'])

        # save declared variables
        self.past_variables = []

        # save declared variable types
        self.past_types = []

        # this will keep track of open closing brackets to try to predict which closing ones are coming next
        self.delim_stack = []

    # dp way of calculating edit distance between two words
    # TODO: end at threshold to save time
    def edit_distance(self, word, target, thresh = 2):
        # table for saved sub solutions
        table = [[0 for _ in range(len(target) + 1)] for _ in range(len(word) + 1)]
        for i in range(len(word) + 1):
            for j in range(len(target) + 1):
                # if left word is empty you have to add the rest of the chars from the right
                if i == 0:
                    table[i][j] = j
                # same as above
                elif j == 0:
                    table[i][j] = i
                # characters are the same so just take the value from the last iteration
                elif word[i - 1] == target[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    table[i][j] = 1 + min(table[i - 1][j], \
                                          table[i][j - 1], \
                                          table[i - 1][j - 1])
        return table[len(word)][len(target)]

    def check_dist(self, token, targets, thresh = 2):
        min_dist = thresh
        closest_word = None
        for t in targets:
            dist = self.edit_distance(token, t, thresh)
            if dist < min_dist:
                min_dist = dist
                closest_word = t
        return closest_word

    def check_operator(self, token):
        if token in self.operators:
            return token
        return None

    def check_str(self, token):
        if token[0] == '"' or token[-1] == '"':
            return True
        return None

    def check_line_term(self, token):
        if token in self.line_terms:
            return token
        return None

    def correct_num(self, token):
        # token is assumed to be a mixture of digits and letters
        # TODO: correct common mistakes made by OCR here. S -> 5, etc.
        if 'S' in token:
            token.replace('S', '5')
            return token
        return None

    def check_num(self, token, thresh = 2):
        # check how many characters are digits
        pos = 0
        for c in token:
            if c.isdigit():
                pos = pos + 1
        # if enough chars are digits, we can assume its a number
        if len(token) - pos < thresh:
            return self.correct_num(token)
        return None

    def check_var(self, token):
        # check if its a variable from the past
        var = self.check_dist(token, self.past_variables)
        if var != None:
            return var
        # check if its a variable
        return token

    def check_literal(self, token):
        s = self.check_str(token)
        if s != None:
            return s
        n = self.check_num(token)
        if n != None:
            return n
        return None

    def tokenize(self, line):
        # split by whitespace
        tokens = line.split()

        # check if its actually two tokens, in this case check for : at the end of line
        if tokens[-1][-1] in self.line_terms:
            end = tokens[-1][-1]
            tokens[-1] = tokens[-1][:-1]
            tokens.append(end)

        # split by operators? what else?

        print(tokens)
        return tokens

    def classify(self, token, thresh = 2):
        # check if its close to a keyword
        kw = self.check_dist(token, self.keywords)
        if kw != None: 
            return kw, 'keyword'

        # check if its an operator
        op = self.check_operator(token)
        if op != None: 
            return op, 'operator'

        # check if end of line indicator
        t = self.check_line_term(token)
        if t != None:
            return t, 'end of line'

        # check if its a literal
        lit = self.check_literal(token)
        if lit != None: 
            return lit, 'literal'

        # check if its a variable
        var = self.check_var(token)
        if var != None:
            return var, 'variable'

        return token, 'unknown'

    def post_process_line(self, line):
        tokens = self.tokenize(line)

        # contains a list of the previous token types
        token_types = []
        line = ""
        for token in tokens:
            word, tt = self.classify(token)
            token_types.append(tt)
            line = line + word + ' '
        return line, token_types