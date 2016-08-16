
warningflags = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wshadow
commoncflags = -std=c++1y -O3 $(warningflags) $(MFLAGS)
AM_CXXFLAGS = $(commoncflags)
AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libnoisegen.la

libnoisegen_la_SOURCES = src/noisegen.cpp
libnoisegen_la_CXXFLAGS = $(commoncflags)
