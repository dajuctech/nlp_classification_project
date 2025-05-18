# Load configuration from YAML
import yaml

def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)