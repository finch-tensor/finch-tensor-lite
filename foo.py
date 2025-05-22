import finch

# Example usage
if __name__ == "__main__":
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = finch.codegen.c.get_c_function("add", c_code)
    result = f(3, 4)
    print("Result:", result)
