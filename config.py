import yaml

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        
    def __getattr__(self, name):
        return self._config.get(name)