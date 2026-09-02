# CMake generated Testfile for 
# Source directory: D:/Personal/Projects/Rayvan/cpp
# Build directory: D:/Personal/Projects/Rayvan/build/emc-cpu
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test("rayvan-emc-native" "D:/Personal/Projects/Rayvan/build/emc-cpu/Debug/rayvan-emc-tests.exe")
  set_tests_properties("rayvan-emc-native" PROPERTIES  _BACKTRACE_TRIPLES "D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;51;add_test;D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test("rayvan-emc-native" "D:/Personal/Projects/Rayvan/build/emc-cpu/Release/rayvan-emc-tests.exe")
  set_tests_properties("rayvan-emc-native" PROPERTIES  _BACKTRACE_TRIPLES "D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;51;add_test;D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test("rayvan-emc-native" "D:/Personal/Projects/Rayvan/build/emc-cpu/MinSizeRel/rayvan-emc-tests.exe")
  set_tests_properties("rayvan-emc-native" PROPERTIES  _BACKTRACE_TRIPLES "D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;51;add_test;D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test("rayvan-emc-native" "D:/Personal/Projects/Rayvan/build/emc-cpu/RelWithDebInfo/rayvan-emc-tests.exe")
  set_tests_properties("rayvan-emc-native" PROPERTIES  _BACKTRACE_TRIPLES "D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;51;add_test;D:/Personal/Projects/Rayvan/cpp/CMakeLists.txt;0;")
else()
  add_test("rayvan-emc-native" NOT_AVAILABLE)
endif()
