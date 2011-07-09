#!/usr/bin/env python2

__doc__ = """
"""

import inspect
import argparse
import cPickle as pickle

import os
import sys


def main(fct, argn = 0, output = False, **descr):
    """
    """
    parser = argparse.ArgumentParser(description = fct.__doc__)
    fct_descr = inspect.getargspec(fct)
    parser.add_argument("--info")

    ndefaults = -len(fct_descr.defaults) if fct_descr.defaults is not None else None
    for arg in fct_descr.args[:argn]:
        parser.add_argument(arg, metavar=arg)

    for arg in fct_descr.args[argn:ndefaults]:
        arg_descr = dict()
        if arg in descr:
            arg_descr['type'] = descr[arg]
        parser.add_argument("--" + arg, required = True, **arg_descr)

    if ndefaults is not None:
        for arg, value in zip(fct_descr.args[ndefaults:], fct_descr.defaults):
            arg_descr = dict()
            if arg in descr:
                arg_descr['type'] = descr[arg]
            parser.add_argument("--" + arg, required = False, default = value, **arg_descr)

    ns = parser.parse_args()

    argv = [getattr(ns, a) for a in fct_descr[0]]

    result = fct(*argv)

    info = " ".join(sys.argv)


import os, sys
from mako.template import Template
from mako.lookup import TemplateLookup

def render(inp, license = "LICENSE"):
    lookupdir = os.path.dirname(inp)
    mylookup = TemplateLookup(directories=[lookupdir])

    with open(license) as f:
        license_text = f.read()

    t = Template(text = open(inp).read(), lookup = mylookup)

    import sys
    sys.stdout.write(t.render(license = license_text))

main(render, argn = 1)

