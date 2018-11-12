"""
********************************
*   Created by mohammed-alaa   *
********************************
Some helper functions for logging and named constants
"""

import sys


def log(*args, file=None):
    """log to a file and console"""
    if file:
        print(*args, file=file)
        file.flush()
    print(*args)
    sys.stdout.flush()


def get_augmenter_text(augmenter_level):
    """augmenter level text"""
    if augmenter_level == 0:
        augmenter_text = "heavy"
    elif augmenter_level == 1:
        augmenter_text = "medium"
    else:  # 2
        augmenter_text = "simple"

    return augmenter_text
