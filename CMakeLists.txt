# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC -O2 -mfloat-abi=softfp -mfpu=neon -Wno-deprecated-register -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -O2 -mfloat-abi=softfp -mfpu=neon -Wno-deprecated-register -fopenmp")

aux_source_directory(src/main/cpp/math_neon MATH_NEON)
add_library(mathneon SHARED  ${MATH_NEON})
target_link_libraries(mathneon m)

set(LIB_NEON_SOURCE
        src/main/cpp/neonAcc.cpp
        src/main/cpp/jni_neon.cpp)

add_library(neontest SHARED  ${LIB_NEON_SOURCE})
target_link_libraries(neontest mathneon)
target_link_libraries(neontest ${log-lib})




