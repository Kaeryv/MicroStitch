cmake_minimum_required(VERSION 3.23)
project(stitcher)
find_package( OpenCV REQUIRED )
add_subdirectory(deps/raylib)
add_subdirectory(deps/json)
add_subdirectory(deps/fmt)
add_executable(stitcher main.cc )
target_link_libraries(stitcher nlohmann_json::nlohmann_json)
target_link_libraries(stitcher raylib)
target_link_libraries(stitcher fmt)
target_link_libraries(stitcher ${OpenCV_LIBS} )
add_compile_options(-fmax-errors=1 -fopenmp)

target_include_directories(stitcher PUBLIC 
	"${PROJECT_SOURCE_DIR}/deps/entt/src/"
	"${PROJECT_SOURCE_DIR}/deps/raygui/src/"
	"${OpenCV_INCLUDE_DIRS}"
)

