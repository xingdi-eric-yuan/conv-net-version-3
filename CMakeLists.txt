
set(the_target "cnn3")
project( ${the_target} )
find_package( OpenCV REQUIRED PATHS /usr/local/opt/opencv3 NO_DEFAULT_PATH)
include_directories( /usr/local/opt/opencv3/include)
aux_source_directory( src/. src_list )
add_executable( ${the_target} ${src_list} )
target_link_libraries (${the_target} ${lstdc++})
target_link_libraries (${the_target} ${OpenCV_LIBS})


