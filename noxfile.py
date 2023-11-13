import nox 
pythonversion = ["3.12"]
@nox.session(python=pythonversion)  
def tests(session):
    session.run("poetry", "install", external=True)  
    session.run("pytest", "--cov")  

@nox.session(python=pythonversion)
def lint(session):
    session.run("poetry", "install", external=True)
    session.run("flake8", "yubo_test" )  
    session.run("flake8", "tests" )


# Coverage.py is a tool for measuring code coverage of Python programs. 
# It monitors your program, noting which parts of the code have been executed, 
# then analyzes the source to identify code that could have been executed but was not.
@nox.session(python=pythonversion)
def coverage(session):
    session.run("poetry", "install", external=True)
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "report")