class AbsAttr:
    """
    A class to represent abstract required attributes.

    ```
        # Example usage:
        class foo(AbsAttr):
            @classmethod
            def __attrs__(cls):
                return ["foo"]

        class bar(foo):
            def __init__(self, qux):
                self.foo = qux

        class zoo(foo):
            def __init__(self, qux):
                self.goo = qux

        bar(42)
        zoo(42) # This will raise an error because 'foo' is not defined in 'zoo'
    ```
    """

    @classmethod
    def __attrs__(cls):
        """
        Returns the attributes that are required for a subclass.
        This method is used to define the attributes that must be present
        in a subclass, ensuring that the class is properly configured.
        """
        return []

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Initializes the subclass and collects the required attributes.
        This method is called when a subclass is created, allowing it to
        gather the attributes that must be present in a Finch model.
        """
        super().__init_subclass__(**kwargs)
        attrs = {}
        for base in reversed(cls.__mro__):
            if hasattr(base, "__attrs__"):
                for attr in base.__attrs__():
                    # Only set if not already set, so the most base class wins
                    if attr not in attrs:
                        attrs[attr] = base
        cls.__init__ = make_validate_attrs_init(cls.__init__, attrs)


def make_validate_attrs_init(init, attrs):
    def __validate_attrs_init__(self, *args, **kwargs):
        """
        Validates the attributes of the class. This method should be implemented
        by subclasses to ensure that the required attributes are present and valid.
        If not implemented, a custom error message will be shown.
        """
        init(self, *args, **kwargs)
        for attr, base in attrs.items():
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclass '{self.__class__.__name__}' of base class"
                    f" '{base.__name__}' is missing required attribute '{attr}'."
                )

    return __validate_attrs_init__
