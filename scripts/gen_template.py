#!/usr/bin/env python2

import os, sys
from mako.template import Template
from mako.lookup import TemplateLookup


inp = sys.argv[1]
out = sys.argv[2]
lic = sys.argv[3] if len(sys.argv) > 3 else 'LICENSE'

lookupdir = os.path.dirname(inp)
mylookup = TemplateLookup(directories=[lookupdir])


with open(lic) as f:
    license_text = f.read()

t = Template(text = open(inp).read(), lookup = mylookup)
with open(out, 'w') as output:
    output.write(t.render(license = license_text))

