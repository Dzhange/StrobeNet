import os
import sys
import shutil
import platform


def startup(cfg, interative=True):
    """
    Before everything begins, prepare preliminary works
    """
    # Check model and metric are saved
    # assert 'model' in cfg.LOGGER_SELECT
    # assert 'metric' in cfg.LOGGER_SELECT
    # Check config
    os_cate = platform.system().lower()
    assert os_cate in ['linux', 'windows']
    print("=" * shutil.get_terminal_size()[0])
    print("Please check the configuration")
    print('-' * shutil.get_terminal_size()[0])
    print(cfg)
    print('-' * shutil.get_terminal_size()[0])
    print('y/n?', end='')
    if interative:
        need_input = True
        while need_input:
            response = input().lower()
            if response in ('y', 'n'):
                need_input = False
        if response == 'n':
            sys.exit()
    else:
        print("Warning, NOT INTERACTIVE CONFIRM!")
    print("=" * shutil.get_terminal_size()[0])

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)[1:-1]
    print("Set GPU: " + str(cfg.GPU)[1:-1] + '...')

    # check log dir
    abs_log_dir = os.path.join(cfg.ROOT_DIR, 'log', cfg.LOG_DIR)
    if cfg.RESUME:
        # if resume, don't remove the old log
        if not os.path.exists(abs_log_dir):
            print("Warning: Need resuem but the log dir: " + abs_log_dir + " doesn't exists, create one", end='')
            os.makedirs(abs_log_dir)
    else:
        if os.path.exists(abs_log_dir):
            print("Warning! Not resume but log dir: " + abs_log_dir + ' already existed. Remove the old log? y/n',
                  end='')
            if interative:
                need_input = True
                while need_input:
                    response = input().lower()
                    if response in ('y', 'n'):
                        need_input = False
                if response == 'n':
                    sys.exit()
            else:
                print("Warning, NOT INTERACTIVE CONFIRM!")
            os.system(
                ("rm " if os_cate == 'linux' else 'rd /s /q ') + abs_log_dir + (" -r" if os_cate == 'linux' else ' '))
        os.makedirs(abs_log_dir)
    print('Log dir confirmed...')

    # save configuration in log dir
    with open(os.path.join(abs_log_dir, 'configuration_%d_ep.txt' % cfg.RESUME_EPOCH_ID), 'w+') as f:
        print(cfg, file=f)
        f.close()
        print('Save configuration to local file...')

    # backup model, dataset, config
    os.system('copy ' if os_cate == 'windows' else 'cp ' + os.path.join(cfg.ROOT_DIR, 'core', 'models',
                                                                        cfg.MODEL + '.py') + ' ' + abs_log_dir)
    os.system('copy ' if os_cate == 'windows' else 'cp ' + os.path.join(cfg.ROOT_DIR, 'inout', 'dataset',
                                                                        cfg.DATASET + '.py') + ' ' + abs_log_dir)
    os.system('copy ' if os_cate == 'windows' else 'cp ' + os.path.join(cfg.ROOT_DIR, 'config', 'config_files',
                                                                        cfg.CONFIG_FILE) + ' ' + abs_log_dir)
    for filename in cfg.BACKUP_FILES:
        os.system('copy ' if os_cate == 'windows' else 'cp ' + os.path.join(cfg.ROOT_DIR, filename) + ' ' + abs_log_dir)
