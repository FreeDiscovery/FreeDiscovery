import sys
import multiprocessing

# Adapter from
# http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input


def _query_yes_no(question, default="yes", overwrite=False):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    if overwrite:
        return overwrite
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


class _TeeLogger(object):
    """ tee stdout & stderr to a file """
    def __init__(self, name, mode='at'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def flush(self):
        self.file.flush()
        self.stdout.flush()
        self.stderr.flush()

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)


def _number_of_workers(max_workers=6):
    """ Compute the number of worker processes for gunicorn"""
    num_w = (multiprocessing.cpu_count() * 2) + 1
    if num_w > max_workers:
        num_w = max_workers
    return num_w
