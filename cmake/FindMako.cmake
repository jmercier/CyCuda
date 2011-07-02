#
# Cython
#

# This finds the "cython" executable in your PATH, and then in some standard
# paths:
FIND_FILE(MAKO_BIN mako-render /usr/bin /usr/local/bin)
FUNCTION(MAKO_ADD_MODULE name)
    SET(infile ${CyCuda_TEMPLATE_DIR}/${name}.mako)
    ADD_CUSTOM_COMMAND(
        OUTPUT ${name}
        COMMAND ${CyCuda_TEMPLATE_COMMAND} ${infile} ${name} ${CyCuda_LICENSE}
        DEPENDS ${infile} ${CyCuda_TEMPLATE_COMMAND}
        COMMENT "Makoing" ${infile}
        )
ENDFUNCTION(MAKO_ADD_MODULE)
