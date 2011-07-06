<%def name="makesection(name)">
#
#
#   ${name}
#
#
</%def>

<%def name="cuda_dealloc(command)">\
def __dealloc__(self):
        cdef CUresult res
        res = ${command}\
<%text>
        if res != CUDA_SUCCESS: print("Error in Cuda deallocation <%s>" % \
                                self.__class__.__name__)
</%text>\
</%def>

#
# vim: filetype=pyrex
#
#
