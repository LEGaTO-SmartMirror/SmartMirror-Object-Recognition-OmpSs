#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "YoloTensorRTWrapper.h"

#define IMAGE_WIDTH  416
#define IMAGE_HEIGHT 416

int main(int argc, char** argv)
{
	char capStr[BUFSIZ];

	sprintf(capStr, "shmsrc socket-path=/dev/shm/camera_small ! video/x-raw, format=BGR, width=%i, height=%i, \
		framerate=30/1 ! queue max-size-buffers=5 leaky=2 ! videoconvert ! video/x-raw, format=BGR ! \
		appsink drop=true",
			IMAGE_WIDTH, IMAGE_HEIGHT);

	if (InitVideoStream(capStr) != 1)
	{
		fprintf(stderr, "Failed to open video sink, exiting ...\n");
		return -1;
	}

	InitYoloTensorRT();

	while (1)
	{
		uint8_t* pData = GetNextFrame();
		ProcessNextFrame(pData, IMAGE_HEIGHT, IMAGE_WIDTH);
		ProcessDetections();

		CheckFPS();
	}

	Cleanup();

	return 0;
}
