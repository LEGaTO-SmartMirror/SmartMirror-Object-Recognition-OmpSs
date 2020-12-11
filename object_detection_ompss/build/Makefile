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
	-I/usr/local/include/opencv4 \
	-I/usr/include/aarch64-linux-gnu \
	-I../src/YoloTensorRT/3rdParty/nvidia \
	-I../src/YoloTensorRT
	#-I/opt/dev/ompss@cluster/bin/oss2cluster/nanos6-AddressSanitizer/include

LIB=-L/usr/local/ -L/usr/local/cuda/lib64/ \
	-L/opt/dev/dependencies/darknet/ \
	-l:libdarknet.so -lcuda -lcudart \
	$(shell pkg-config opencv4 --libs) \
	-L/usr/lib/aarch64-linux-gnu -lnvinfer -lnvparsers -lnvonnxparser -L../src/YoloTensorRT -lYoloLayerPlugin # TensorRT
	#-L/opt/dev/ompss@cluster/bin/oss2cluster/nanos6-AddressSanitizer/lib/


#CFLAGS = -ggdb -c    # -fsanitize=address -fno-omit-frame-pointer
#CXXFLAGS = -ggdb -c  # -fsanitize=address -fno-omit-frame-pointer
#MCCFLAGS = -ggdb -c  --ompss-2 -k  # -fsanitize=address -fno-omit-frame-pointer

CFLAGS = -c -O3 -march=native -mtune=native -DNDEBUG
CXXFLAGS =  -c -O3 -march=native -mtune=native -DNDEBUG
MCCFLAGS =  -c --ompss-2

LINKFLAGS = --ompss-2
CPPSOURCES= $(wildcard ../src/*.cpp)
CPPSOURCES_NAMES = $(notdir $(CPPSOURCES))
CPPOBJECTS = $(CPPSOURCES_NAMES:%.cpp=%.o)
OBJECTS = $(wildcard *.o)
MCCFILES = $(wildcard mcc_*)

EXECUTABLE = objDetect.bin
YoloTRT = YoloTRT.bin

SYNC  = scp ./${EXECUTABLE}  $(USER)@smartmirror5-2:/opt/dev/MagicMirror/modules/SmartMirror-Object-Recognition-OmpSs/object_detection_ompss/build/
SYNC2 = scp ./${YoloTRT}     $(USER)@smartmirror5-2:/opt/dev/MagicMirror/modules/SmartMirror-Object-Recognition-OmpSs/object_detection_ompss/build/

all: $(EXECUTABLE) $(YoloTRT)

$(EXECUTABLE):  Hungarian.o $(CPPOBJECTS) MYSort.o main.o
	$(MCXX) $(LINKFLAGS) $^ $(LIB) -o $@
	$(CPY)
	$(SYNC)

$(YoloTRT): main_TRT.o YoloWrapper.o
	$(MCXX) $(LINKFLAGS) $^ $(LIB) -o $@
	$(SYNC2)

main.o: ../src/main.c
	$(MCC) $(MCCFLAGS) $(INC) $< -o $@

main_TRT.o: ../src/main_TRT.c
	$(MCC) $(MCCFLAGS) $(INC) $< -o $@

Hungarian.o: ../src/Hungarian.c
	$(CC) $(CFLAGS) $(INC) $< -o $@

MYSort.o: ../src/MYSort.c
	$(CC) $(CFLAGS) $(INC) $< -o $@

YoloWrapper.o: ../src/YoloTensorRT/YoloTensorRTWrapper.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -Wno-deprecated-declarations $(INC) $< -o $@

%.o: ../src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(CPPOBJECTS) $(MCCFILES) $(EXECUTABLE)
