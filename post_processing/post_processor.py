
""" 

1. score each token based on what we think it is, adjust for each token
2. read all tokens and categorize them, adjust for entire line

local : fix single token based on other parts of same token
    - predict what each token is based on characters. BEGIN/END are important
    - 
global: fix single token based on surrounding tokens

DONE:
    - correct key words
    - correct types

TODO:
    - use past tokens to help with next
    - if last token: do ___
    - account for ( being part of the beginning of word
    - _ vs. -
    - || vs ll vs II
    - help predict [ vs { vs (
    - predict ; (can the first token tell us if there will be ; at the end?)

"""

class Post_Processor(object):
    def __init__(self, language = 'C'):
        if language == 'C':
            import sys
            from C.c_post_utils import C_Post_Processor
            self.processor = C_Post_Processor()
        elif language == 'Python':
            from Python.python_post_utils import Python_Post_Processor
            self.processor = Python_Post_Processor()
        else:
            raise Exception("{} post processor not available".format(language))
            quit(1)


    def process_line(self, line):
        return self.processor.post_process_line(line)

    def process_lines(self, lines):
        if type(lines) is list:
            output = ""
            for l in lines:
                output = output + self.processor.post_process_line(l) + '\n'
            return output
        elif type(lines) is str:
            output = ""


if __name__ == '__main__':
    pp = Post_Processor('Python')
    line = pp.process_line("fOr x In y:")
    print(line)



