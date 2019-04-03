
add_library(sobol_c STATIC sobol_c.cpp)

add_executable(sobol_test main.cpp)
target_link_libraries(sobol_test sobol_c)

