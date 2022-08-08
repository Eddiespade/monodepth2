#!/usr/bin/python
import os, sys

path = "../../data/2022_07_04_02/image_02/data/"
# path = "../../data/2022_07_04_01/deep/"


def rename(path):
    for name in os.listdir(path):
        newName = name.replace("2_", "0000")
        if not name.startswith("2_"):
            os.remove(os.path.join(path, name))
        else:
            os.rename(os.path.join(path, name), os.path.join(path, newName))


rename(path)
