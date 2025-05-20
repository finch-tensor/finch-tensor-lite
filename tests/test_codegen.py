import finch


def test_add_function():
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = finch.codegen.c.get_c_function("add", c_code)
    result = f(3, 4)
    assert result == 7, f"Expected 7, got {result}"
