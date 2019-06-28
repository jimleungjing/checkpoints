import os
import subprocess

IMG_DIR = 'images'

if __name__ == '__main__':
    for root, dirs, files in os.walk(IMG_DIR, topdown=False):
        for name in files:
            # print(os.path.splitext(name)[0])
            print(os.path.join(root, name))