#
# Cython
#

# This finds the "cython" executable in your PATH, and then in some standard
# paths:
#

MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name})
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} FALSE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    SET(larg_names ${arg_names})
    LIST(FIND larg_names "${arg}" is_arg_name)
    IF (is_arg_name GREATER -1)
      SET(${prefix}_${current_arg_name} ${current_arg_list})
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSE (is_arg_name GREATER -1)
      SET(loption_names ${option_names})
      LIST(FIND loption_names "${arg}" is_option)
      IF (is_option GREATER -1)
             SET(${prefix}_${arg} TRUE)
      ELSE (is_option GREATER -1)
             SET(current_arg_list ${current_arg_list} ${arg})
      ENDIF (is_option GREATER -1)
    ENDIF (is_arg_name GREATER -1)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)

FIND_FILE(MAKO_BIN mako-render /usr/bin /usr/local/bin)
SET(MAKO_TEMPLATE_COMMAND ${MAKO_BIN})
SET(MAKO_TEMPLATE_DIR "")
SET(MAKO_EXTRA_ARGS "")

FUNCTION(MAKO_ADD_MODULE name)
    PARSE_ARGUMENTS(mako "TEMPLATE;EXTRA_ARGS" "" ${ARGN})
    IF (${mako_EXTRA_ARGS})
    ELSE (${mako_EXTRA_ARGS})
        set(mako_EXTRA_ARGS ${MAKO_EXTRA_ARGS})
    ENDIF (${mako_EXTRA_ARGS})
    IF (${mako_TEMPLATE})
    ELSE (${mako_TEMPLATE})
        set(mako_TEMPLATE ${MAKO_TEMPLATE_DIR}/${name}.mako)
    ENDIF (${mako_TEMPLATE})
    ADD_CUSTOM_COMMAND(
        OUTPUT ${name}
        COMMAND ${MAKO_TEMPLATE_COMMAND} ${mako_TEMPLATE} ${mako_EXTRA_ARGS} > ${name}
        DEPENDS ${mako_TEMPLATE} ${MAKO_TEMPLATE_COMMAND}
        COMMENT "Makoing" ${infile}
        VERBATIM
        )
ENDFUNCTION(MAKO_ADD_MODULE)
