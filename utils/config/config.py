import json


class JsonObj:
    def __init__(self, d):
        self.__dict__ = d


def make_config(cfg_file):
    f_config = open(cfg_file, 'r')
    cfg = json.loads(f_config.read(), object_hook=JsonObj)
    f_config.close()
    print('\n|      '.join(['%s:%s' % item for item in cfg.__dict__.items()]))
    return cfg
