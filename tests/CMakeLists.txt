# --------------------------------------------------------------------------
#
# Copyright (C) 2013 - 2016 by the adaflo authors
#
# This file is part of the adaflo library.
#
# The adaflo library is free software; you can use it, redistribute it,
# and/or modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.  The full text of the
# license can be found in the file LICENSE at the top level of the adaflo
# distribution.
#
# --------------------------------------------------------------------------

########### This file is based on the respective code in the deal.II test
########### suite. The mechanism to filter out lines and detect the number of
########### MPI procs in the input file was inspired by ASPECT,
########### https://aspect.dealii.org

MACRO(SET_IF_EMPTY _variable _value)
  IF("${${_variable}}" STREQUAL "")
    SET(${_variable} ${_value})
  ENDIF()
ENDMACRO()

# A function that extracts the number of MPI processes from an input file
# This is encoded in .prm files through lines of the form
#    '# mpirun: 4'
# The result is returned in a variable _mpi_count
FUNCTION(get_mpi_count _filename)
  FILE(STRINGS ${_filename} _input_lines
       REGEX "mpirun:")
  IF("${_input_lines}" STREQUAL "")
    SET(_mpi_count 1 PARENT_SCOPE)
  ELSE()
    # go over the (possibly multiple) lines with MPI markers and choose the last
    FOREACH(_input_line ${_input_lines})
     SET(_last_line ${_input_line})
    ENDFOREACH()
    STRING(REGEX REPLACE "^ *# *mpirun: *([0-9]+) *$" "\\1"
           _mpi_count ${_last_line})
    SET(_mpi_count "${_mpi_count}" PARENT_SCOPE)
  endif()
ENDFUNCTION()


# time limit of 10 minutes per test
SET_IF_EMPTY(TEST_TIME_LIMIT 600)

############################3

ADD_CUSTOM_TARGET(tests)

#
# Select diff tool as provided by deal.II
#

FIND_PROGRAM(DIFF_EXECUTABLE
  NAMES diff
  HINTS ${DIFF_DIR}
  PATH_SUFFIXES bin
  )

FIND_PROGRAM(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )

MARK_AS_ADVANCED(DIFF_EXECUTABLE NUMDIFF_EXECUTABLE)

IF("${TEST_DIFF}" STREQUAL "")
  IF(NOT NUMDIFF_EXECUTABLE MATCHES "-NOTFOUND")
    SET(TEST_DIFF ${NUMDIFF_EXECUTABLE} -a 1e-5 -r 1e-8 -s ' \\t\\n:,')
    IF(DIFF_EXECUTABLE MATCHES "-NOTFOUND")
      SET(DIFF_EXECUTABLE ${NUMDIFF_EXECUTABLE})
    ENDIF()
  ELSEIF(NOT DIFF_EXECUTABLE MATCHES "-NOTFOUND")
    SET(TEST_DIFF ${DIFF_EXECUTABLE})
  ELSE()
    MESSAGE(FATAL_ERROR
      "Could not find diff or numdiff. One of those are required for running the tests.\n"
      "Please specify TEST_DIFF by hand."
      )
  ENDIF()
ENDIF()


# The tests are set up in two layers. There are several executables sitting in
# the tests subdirectory (found by the GLOB immediately following here). Then,
# each executable can be run in various combinations, controlled by an input
# file *.prm. To detect the input files, the base name of the file must match
# the *.cc file but additional specifiers can be present afterwards (this is
# done by the GLOB a few lines further down).
FILE(GLOB _testexes *.cc)

SET(_n_tests "0")
LIST(SORT _testexes)
FOREACH(_testexe ${_testexes})
  GET_FILENAME_COMPONENT(_testexe ${_testexe} NAME_WE)

  ADD_EXECUTABLE( ${_testexe} ${_testexe}.cc)
  DEAL_II_SETUP_TARGET(${_testexe})
  TARGET_LINK_LIBRARIES(${_testexe} adaflo)

  FILE(GLOB _tests ${_testexe}*.prm)
  LIST (SORT _tests)
  FOREACH(_test ${_tests})
    GET_FILENAME_COMPONENT(_test ${_test} NAME_WE)

    MATH(EXPR _n_tests "${_n_tests} + 1")

    # Create the output directory and a rule to generate the input file.  This
    # input file is the one found in the source dir, but it has the correct
    # output directory so that we do not need to specify it by hand
    FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output-${_test})

    # then generate a rule that runs the test executable and normalizes the
    # output files. Before running, delete prior content of the directory to
    # make sure no dead files are left there. we have to take care of not
    # deleting those files that have been placed there on purpose, however,
    # which are all of the .cmp.notime files.
    GET_MPI_COUNT(${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm)
    ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
      COMMAND
        for i in ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/* \; do
          if echo \$i | grep -q -v .cmp.notime \; then
            rm -f \$i \;
          fi \;
        done
      COMMAND mpirun -np ${_mpi_count} ${CMAKE_CURRENT_BINARY_DIR}/${_testexe}
              ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm
              > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output.tmp
      COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output.tmp
                 ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.prm ${_testdepends} ${_testexe}
      )

    # The final target for this test
    ADD_CUSTOM_TARGET(${_test}.run
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
      )

    ADD_CUSTOM_TARGET(tests.${_test})


    # create the output.notime target, and also create a similar file for the
    # screen output. the test_name.output files are typically copied directly
    # from a run but the timer information is obviously not reproducible and
    # needs to be stripped out. This is done by the two egrep commands, one
    # for the final timer output and the second for the in-code statistics
    # that might be present, respectively. The two lines involving sed are
    # used to process output like [12.3/7] (which is a typical solver output
    # of a residual and the linear iterations) where we want to insert a bit
    # of whitespace around the residual in order to enable numdiff to identify
    # that as numbers and apply the round-off rules, and we want to eliminate
    # the iteration count because that can be slightly different (+- 2
    # iterations) on different systems
    #
    # we then do the same thing also for the screen out
    ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
      COMMAND
      cat ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.output
          | egrep -v '^\\|'
          | egrep -v '^--'
          | sed 's/\\[/\\[ /g'
          | sed 's/\\/[0-9]*\\]/ \\]/g'
          | sed 's/\\// \\//g'
          > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.output
      )

    ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
      COMMAND
      cat ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
          | egrep -v '^\\|'
          | egrep -v '^--'
          | sed 's/\\[/\\[ /g'
          | sed 's/\\/[0-9]*\\]/ \\]/g'
          | sed 's/\\// \\//g'
          > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/screen-output
              ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
      )

    # create the target that compares the .notime with the saved file
    ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
      COMMAND
          if (${TEST_DIFF} ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
                ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
              > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff) \; then
            : \;
          else
            mv ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff
               ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
            echo "******* Error during diffing output results for ${_test}/${_test}" \;
            echo "******* Results are stored in ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed" \;
            echo "******* Check: ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}/${_test}" \;
            echo "******* Diffs are:" \;
            cat ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
            false \;
          fi
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.cmp.notime
              ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.notime
      )

    # add the target for this output file to the dependencies of this test
    ADD_CUSTOM_TARGET(${_test}.diff
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
      )

    ADD_DEPENDENCIES(tests ${_test}.diff)
    ADD_TEST(NAME ${_test}
      COMMAND
      ${CMAKE_COMMAND}
      -DBINARY_DIR=${CMAKE_BINARY_DIR}
      -DTESTNAME=${_test}
      -DERROR="Test ${_test} failed"
      -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
    SET_TESTS_PROPERTIES(${_test} PROPERTIES
      TIMEOUT ${TEST_TIME_LIMIT}
      )
  ENDFOREACH()
ENDFOREACH()

MESSAGE(STATUS "Added ${_n_tests} tests")
