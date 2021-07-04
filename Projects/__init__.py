import imath as im
import yaml
import os


def get_path(path, yaml_path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(os.path.split(yaml_path)[0], path)

def load_from_source(source, name):
    env = {}
    exec(open(source, 'r').read(), env)
    return env[name]


def train(train_yaml_file):
    train_cfg = yaml.load(open(train_yaml_file, 'r').read())

    print(train_cfg)

    # load dataloader
    assert 'dataloader' in train_cfg.keys()
    dataloader_cfg = train_cfg['dataloader']
    source = get_path(dataloader_cfg['source'], train_yaml_file)

    dataloader_class = load_from_source(source, dataloader_cfg['name'])
    dataloader = dataloader_class(**dataloader_cfg['params'])

    print(dataloader)

    # load trainer
    assert 'trainer' in train_cfg.keys()
    trainer_cfg = train_cfg['trainer']
    source = get_path(trainer_cfg['source'], train_yaml_file)

    trainer_class = load_from_source(source, trainer_cfg['name'])
    trainer = trainer_class(**trainer_cfg['params'])

    print(trainer)
    params = trainer_cfg['params']
    model = params['model']

    print(model)




if __name__ == "__main__":
    train("./tmp.yaml")
