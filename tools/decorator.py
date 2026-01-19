import inspect
from .base import Tool

def tool(name: str = None, description: str = None):
    def wrapper(func):
        """
        A decorator that creates a Tool instance from the given function.
        """
        # Get the function signature
        signature = inspect.signature(func)

        # Extract (param_name, param_annotation) pairs for inputs
        arguments = []
        for param in signature.parameters.values():
            annotation_name = (
                param.annotation.__name__
                if hasattr(param.annotation, '__name__')
                else str(param.annotation)
            )
            arguments.append((param.name, annotation_name)) # so your could Annoutation

        # Determine the return annotation
        return_annotation = signature.return_annotation
        if return_annotation is inspect._empty:
            outputs = "No return annotation"
        else:
            outputs = (
                return_annotation.__name__
                if hasattr(return_annotation, '__name__')
                else str(return_annotation)
            )

        func_description = description or func.__doc__ or "No description provided."
        func_name = name or func.__name__

        return Tool(
            name=func_name,
            description=func_description,
            func=func,
            arguments=arguments,
            outputs=outputs,
        )
    return wrapper
