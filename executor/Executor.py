import subprocess
import os

class Executor(object):
    supportedLanguages = [ 'Python', 'C' ]
    languageCommands = { 'Python' : 'python2' }
    languageFileExt = { 'Python' : 'py', 'C': 'c', 'C++': 'cpp' }

    def __init__(self, language='C'):
        if language in self.supportedLanguages:
            self.language = language
        else:
            raise Exception("{} executor not available".format(language))
            quit(1)

    def run(self, program):
        tempFileName = 'temp.' + self.languageFileExt[self.language]

        tempFile = open(tempFileName,"w+")
        tempFile.write(program)
        tempFile.close()

        output = None
        status = 0
        # Fork and Exec, get output + status
        try:
            output = subprocess.check_output([self.languageCommands[self.language], tempFileName])
        except subprocess.CalledProcessError as e:
            output =  e.output
            status = e.returncode
        except:
            output = 'Error executing program.'
            status = 1

        os.remove(tempFileName)
        return output, status

        #
