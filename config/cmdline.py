import argparse


def merge_from_cmdline(cfg):
    """
    Merge some usually changed setting from comand line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='None', help="Choose Config file")
    parser.add_argument('--gpu', type=str, default=None, help="Set GPU device, type=str, e.g: --gpu=0,1,2 ")
    parser.add_argument('--logdir', type=str, default=None, help='log dir name in $project/log/XXXX/.... e.g. exp')
    # parser.add_argument('--resume', type=int, default=None,
    #                     help="resume config: from last time: -1, from best ever: -2, from x epcho: x")
    cmd = vars(parser.parse_args())  # use as dict
    if cmd['config'] is not 'None':
        cfg.CONFIG_FILE = cmd['config']
    if cmd['gpu'] is not None:
        cfg.GPU = [int(id) for id in cmd['gpu'].split(",")]
    if cmd['logdir'] is not None:
        cfg.LOG_DIR = cmd['logdir']
    # if cmd['resume'] is not None:
    #     cfg.resume = cmd['resume']
    return cfg
