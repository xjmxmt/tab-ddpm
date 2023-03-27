import sys, os, time


class Logger(object):
    def __init__(self, parent_dir, ds_name, encoding_type, decoding_type, comment=None, stream=sys.stdout):
        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        if comment is not None:
            log_name = ds_name + "-" + encoding_type + "-" + decoding_type + "-" + comment + "-" + log_name_time + ".txt"
        else:
            log_name = ds_name + "-" + encoding_type + "-" + decoding_type + "-" + log_name_time + ".txt"
        filename = os.path.join(parent_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')
        self.fileno = lambda: False

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass