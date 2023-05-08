def str2bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise KeyError(value)
