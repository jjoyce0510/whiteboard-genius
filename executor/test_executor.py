from executor import Executor

executor = Executor('Python')
goodProgram = 'print "This is a good program"\nexit(0)'
badProgram = 'print "This is a bad program"\nexit(1)'
output, status = executor.run(goodProgram)
expectedStatus = 0
if status is not expectedStatus:
    print "Assertion Error: Expected {} but found {}.".format(expectedStatus, status)
    exit(1)

output, status = executor.run(badProgram)
expectedStatus = 1
if status is not expectedStatus:
    print "Assertion Error: Expected {} but found {}.".format(expectedStatus, status)
    exit(1)

print "Executor Tests Passed!"
exit(0)
