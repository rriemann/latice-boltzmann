TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

QMAKE_LFLAGS   += -std=c++11
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -Wold-style-cast -Wall -Wuninitialized -Wextra
QMAKE_CXXFLAGS_RELEASE += -funroll-loops # unroll short, iterative for-loops
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS_DEBUG   += -O0

contains( QMAKE_CC, gcc ) {
    # http://gcc.gnu.org/bugzilla/show_bug.cgi?id=55805
    QMAKE_CXXFLAGS += -Wno-missing-field-initializers

    # enable openmp
    LIBS += -lgomp
    QMAKE_CXXFLAGS += -fopenmp
}
