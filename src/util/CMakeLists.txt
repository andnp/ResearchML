add_library(logger Logger/Logger.cpp)
add_library(random Random/rand.cpp)
add_library(jsonwrapper json.cpp)
add_library(files Files/files.cpp)
add_library(cdash cdash.cpp)

set(UTILS logger random jsonwrapper files cdash PARENT_SCOPE)
