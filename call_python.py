#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import re
import logging
from subprocess import CalledProcessError, Popen, PIPE, STDOUT
try:
    from ConfigParser import SafeConfigParser
except ModuleNotFoundError:
    from configparser import SafeConfigParser

op = os.path

class Main(object):
    def __init__(self, script=None, *args):
        if script is None:
            logging.error("No Python script supplied.")
            sys.exit(2)
        self.script = script
        self.args = list(args)
        self.root_dir = op.dirname(op.realpath(__file__))
        config = SafeConfigParser()
        with open(op.join(self.root_dir, 'config.cfg')) as cfg_file:
            config.readfp(cfg_file)
        config.read(op.join(self.root_dir, "config.mine.cfg"))
        self.config = config
        self.os = self._os()

    def _os(self):
        if sys.platform.startswith("win"):
            return "win"
        if sys.platform.startswith("darwin"):
            return "mac"
        return "linux"

    @property
    def win(self):
        return self.os == "win"

    @property
    def mac(self):
        return self.os == "mac"

    @property
    def linux(self):
        return self.os == "linux"

    def wait(self):
        if self.win:
            os.system("pause")
        else:
            os.system("/bin/bash -c \"read -s -n 1 -p 'Press any key to continue...'\"")
            print()

    def get(self, name):
        return self.config.get(self.os, name)

    def pop_flag(self, flag):
        try:
            self.args.remove(flag)
            return True
        except ValueError:
            pass
        return False

    def main(self):
        wait = self.pop_flag('-w') or self.pop_flag('-wait')
        direct = self.pop_flag('--direct-stdout')
        if '-v' in self.args or '-verbose' in self.args:
            logging.getLogger().setLevel(logging.INFO)
        python = self.get("python")
        logging.info("Python binary: %s", python)
        script = self.script
        if script not in ("-c", "-m"):
            script = op.join(self.root_dir, script)
            logging.info("Script to execute: %s", script)
        flags = " ".join(self.args)
        cmd = "{python} {cmd} {flags}".format(python=python, cmd=script, flags=flags)
        logging.info("Full command: %s", cmd)
        if direct:
            sys.exit(os.system(cmd))
        exit_code = 0
        try:
            proc = Popen(cmd, cwd=self.root_dir, shell=True, stdout=PIPE, stderr=STDOUT)
            for line in iter(proc.stdout.readline, b''):
                line_text = line.rstrip(b'\r\n')
                line_text_str = line_text.decode('utf-8')
                if line_text == line:
                    print(line_text_str, file=sys.stderr, end='')
                else:
                    print(line_text_str, file=sys.stderr)
                sys.stderr.flush()
                if re.match(r'FAILED \(.*\)$', line_text_str):
                    exit_code = 1
            proc.communicate()
            if proc.returncode:
                exit_code = 1
        except CalledProcessError as e:
            print(e)
            exit_code = 1
        finally:
            if wait:
                self.wait()
        sys.exit(exit_code)


if __name__ == "__main__":
    Main(*sys.argv[1:]).main()
