add_library(logger Logger/pack.cpp Logger/Logger.cpp)
add_library(random Random/rand.cpp)
add_library(jsonwrapper json.cpp)
add_library(files Files/files.cpp)

set(UTILS logger random jsonwrapper files PARENT_SCOPE)
