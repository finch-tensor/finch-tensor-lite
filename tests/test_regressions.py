"""
Tests to check if regression fixtures work as expected.
"""


def test_c_program(file_regression):
    """
    Example test for a C program using the program_regression fixture.
    This test will generate a program and compare it against the stored regression.
    """
    # Your C program logic here
    program = 'int main() { int a= 5; printf("Hello, World!"); return 0; }'
    # Compare the generated program against the stored regression
    file_regression.check(program, extension=".c")


def test_file_regression(file_regression):
    content = "This is a test file content."
    file_regression.check(content)
