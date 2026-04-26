import importlib
from typing import Any


def get_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def convert_value(value: Any):
    if isinstance(value, dict):
        return {k: convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)

            float_val = float(value)
            if float_val.is_integer() and "." not in value:
                return int(float_val)
            return float_val
        except ValueError:
            return value
    return value
