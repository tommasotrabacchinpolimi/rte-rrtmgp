#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles "../src/*.cpp")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp STATIC ${sourcefiles} Aerosol_optics.cpp ../include/Aerosol_optics.h)
target_link_libraries(rte_rrtmgp rte_rrtmgp_kernels)
