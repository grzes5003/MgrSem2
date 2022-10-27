cmake_minimum_required(VERSION 3.16.3)
project(LAB02 C)
set(CMAKE_C_STANDARD 99)
find_package(MPI)

add_executable(LAB02 main.c sieve.c sieve.h)

if(MPI_COMPILE_FLAGS)
    set_target_properties(LAB02 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(LAB02 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(LAB02 m ${MPI_C_LIBRARIES})
