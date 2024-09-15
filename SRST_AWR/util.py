import os
import shutil
import sys
from datetime import datetime

class myLogger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(os.path.join(self.path, "log_verb.txt"), 'a') as f:
            f.write(msg + "\n")

def backup_codes(dest_dir):
    command_file = os.path.join('experiments', 'launch_command.txt')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(command_file, 'a') as f:
        f.write(f'Launch Time: {current_time}\n')
        f.write(' '.join(sys.argv) + '\n\n')
        print(f'Saved launch command and time to {command_file}')

    src_dirs = ['.', 'attacks', 'autoattack', 'load_data', 'models', 'utils']
    for src_dir in src_dirs:
        if not os.path.exists(os.path.join(dest_dir, 'codes', src_dir)):
            os.makedirs(os.path.join(dest_dir, 'codes', src_dir))
        for item in os.listdir(src_dir):
            if item.endswith('.py'):
                src_file = os.path.join(src_dir, item)
                dest_file = os.path.join(dest_dir, 'codes', src_dir, item)
                shutil.copy(src_file, dest_file)
                print(f'Copied: {src_file} to {dest_file}')
