# author: Robert Simari
# date  : 4/23/18
# functions designed for doing post processing for C programs


class C_Post_Processor(object):
    def __init__(self):
        # C Keywords
        self.types = ["short", "int", "long", "float", "double", "char", "void", "bool", "FILE"]
        self.containers = ["enum", "struct", "union", "typedef"]
        self.modifiers = [ "const", "volatile", "extern", "static", "register", "signed", "unsigned"]
        self.flow = [ "if", "else",
                 "goto",
                 "case", "default",
                 "continue", "break"]
        self.loops = ["for", "do", "while", "switch"]
        self.keywords = self.types + self.containers + self.modifiers + self.flow + self.loops + [ "return", "sizeof" ]
        self.prefix_operations = ["-","+","*","&","~","!","++","--"]
        self.postfix_operations = ["++", "--"]
        self.selection_operations = [".","->"] 
        self.multiplication_operations = ["*","/","%"] 
        self.addition_operations = ["+","-"] 
        self.bitshift_operations = ["<<",">>"] 
        self.relation_operations = ["<","<=",">",">="] 
        self.equality_operations = ["==","!="] 
        self.bitwise_operations = ["&", "^", "|"] 
        self.logical_operations = ["&&","||"]
        self.ternary_operations = ["?",":"]

        self.assignment_operations = ["=",
                                "+=","-=",
                                "/=","*=","%="
                                "<<=",">>=",
                             "&=","^=","|=",
                            ]
        self.binary_operations = self.multiplication_operations + \
                            self.addition_operations + \
                            self.bitshift_operations + \
                            self.relation_operations + \
                            self.equality_operations + \
                            self.bitwise_operations  + \
                            self.logical_operations  + \
                            self.assignment_operations + self.selection_operations

        self.operators = self.prefix_operations + self.binary_operations + self.ternary_operations
        self.precedence = [
            self.selection_operations,
            self.multiplication_operations,
            self.addition_operations,
            self.bitshift_operations,
            self.relation_operations,
            self.equality_operations,
            ["&"],["^"],["|"],
            self.logical_operations,
            self.ternary_operations,
            self.assignment_operations,
        ]

        self.delimiters = [' ','+','-','*','/',',',';','>','<','=','(',')','[',']','{','}']

        self.brackets = ['(',')','[',']','{','}']

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
        if token in operators:
            return token
        return None

    def check_str(self, token):
        if token[0] == '"':
            return True

    def correct_num(self, token):
        # token is assumed to be a mixture of digits and letters
        # TODO: correct common mistakes made by OCR here. S -> 5, etc.
        return token.replace('S', '5')

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

    def check_literal(self, token):
        s = self.check_str(token)
        if s != None:
            return s
        n = self.check_num(token)
        if n != None:
            return n

    def classify(self, token, thresh = 2):
        # check if its close to a keyword
        kw = self.check_dist(token, self.keywords)
        if kw != None: 
            return kw, 'keyword'

        # check if its an operator
        op = self.check_operator(token)
        if op != None: 
            return op, 'operator'

        # check if its a literal
        lit = self.check_literal(token)
        if lit != None: 
            return lit, 'literal'

        # check if its a variable from the past
        var = self.check_dist(token, self.past_variables)
        if var != None: 
            past_variables.append(var)
            return var, 'variable'

        return token, 'unknown'

    def post_process_line(self, line):
        # THIS IS A HUGE ASSUMPTION THAT EACH TOKEN IS SPLIT BY ' '
        tokens = line.split()

        # contains a list of the previous token types
        token_types = []
        line = ""
        for token in tokens:
            word, tt = self.classify(token)
            token_types.append(tt)
            line = line + word + ' '
        return line, token_types