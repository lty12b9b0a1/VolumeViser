
option(ENABLE_VIDEO_MAKER "Enable VolRenderer output in video format" OFF)
option(BUILD_VolRenderer_UI "" ON)

file(GLOB_RECURSE
        VolRenderer_SRCS
        "src/*.hpp"
        "src/*.cpp")

add_executable(VolRenderer ${VolRenderer_SRCS})

target_link_libraries(VolRenderer
        PRIVATE
        VolumeViserLib)

target_compile_features(VolRenderer PRIVATE cxx_std_20)

if(ENABLE_VIDEO_MAKER)
    target_compile_definitions(VolRenderer PRIVATE BUILD_WITH_VIDEO_MAKER)

endif()

if(BUILD_VolRenderer_UI)
    add_subdirectory(ui)
endif()