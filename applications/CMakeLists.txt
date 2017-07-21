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

PROJECT(adaflo_examples)

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        )

SET( SOURCE_FILES
     micro_particle.cc
     micro_solver_phase_field.cc
     drivencavity.cc
     periodic_channel.cc
     rising_bubble_parallelpaper.cc
     )

FOREACH ( sourcefile ${SOURCE_FILES} )
        # string replace: cut off .cc from files
        STRING( REPLACE ".cc" "" testname ${sourcefile} )
        ADD_EXECUTABLE( ${testname} ${sourcefile})
        DEAL_II_SETUP_TARGET(${testname})
        TARGET_LINK_LIBRARIES(${testname} adaflo)
ENDFOREACH ( sourcefile ${APP_SOURCES} )
