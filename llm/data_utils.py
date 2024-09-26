import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(path: str) -> str:
    with open(path, "r") as file:
        data = file.read()
    return data