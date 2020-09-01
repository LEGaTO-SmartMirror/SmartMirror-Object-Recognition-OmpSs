## target: dependancy
# recipe of how to use the dependancy to generate the target
#ASAN_OPTIONS=disable_coredump=0,abort_on_error=1

CC = gcc
CXX = g++
MCXX = mcxx
MCC = mcc

NANOS6_VARIANT = nanos6

# -AddressSanitizer

DRKNET=ompss@cluster

 # dependencies

INC=-I/opt/dev/dependencies/darknet/include \
	-I/opt/dev/dependencies/darknet/src \
	-I/usr/local/cuda/include \
	-I/usr/local/include/opencv4
	#-I/opt/dev/ompss@cluster/bin/oss2cluster/nanos6-AddressSanitizer/include

LIB=-L/usr/local/ -L/usr/local/cuda/lib64/ \
	-L/opt/dev/dependencies/darknet/ \
	-l:libdarknet.so -lcuda -lcudart \
	$(shell pkg-config opencv4 --libs)  #  -lasan
	#-L/opt/dev/ompss@cluster/bin/oss2cluster/nanos6-AddressSanitizer/lib/


#CFLAGS = -ggdb -c    # -fsanitize=address -fno-omit-frame-pointer
#CXXFLAGS = -ggdb -c  # -fsanitize=address -fno-omit-frame-pointer
#MCCFLAGS = -ggdb -c  --ompss-2 -k  # -fsanitize=address -fno-omit-frame-pointer

CFLAGS = -c
CXXFLAGS =  -c
MCCFLAGS =  -c --ompss-2

LINKFLAGS = --ompss-2
CPPSOURCES= $(wildcard ../src/*.cpp)
CPPSOURCES_NAMES = $(notdir $(CPPSOURCES))
CPPOBJECTS = $(CPPSOURCES_NAMES:%.cpp=%.o)
OBJECTS = $(wildcard *.o)
MCCFILES = $(wildcard mcc_*)

EXECUTABLE = objDetect.bin

SYNC = scp ./${EXECUTABLE}  $(USER)@smartmirror4:/opt/dev/SmartMirror-Object-Recognition-OmpSs/object_detection_ompss/build/

all: $(EXECUTABLE)

$(EXECUTABLE):  Hungarian.o $(CPPOBJECTS) MYSort.o main.o
	$(MCXX) $(LINKFLAGS) $^ $(LIB) -o $@
	$(CPY)
	$(SYNC)

main.o: ../src/main.c
	$(MCC) $(MCCFLAGS) $(INC) $< -o $@

Hungarian.o: ../src/Hungarian.c
	$(CC) $(CFLAGS) $(INC) $< -o $@

MYSort.o: ../src/MYSort.c
	$(CC) $(CFLAGS) $(INC) $< -o $@

%.o: ../src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(CPPOBJECTS) $(MCCFILES) $(EXECUTABLE)
