#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>

#include "common.h"


#include <nanos6/debug.h>
#include <nanos6/cluster.h>

//#define CLASSES         80 // 80 item in coco.ppNames file 
#define THRESH		.4
#define HIER_THRESH	.4
#define CHANNELS	3
#define IMAGE_WIDTH	416
#define IMAGE_HEIGHT	416
#define CFG_FILE	"../data/enet-coco.cfg"
#define WEIGHT_FILE	"../data/enetb0-coco_final.weights"
#define NAMES_FILE	"../data/coco.names"

int main(int argc, char** argv)
{
	/* ===== Variable Declarations ===== */
	/*  ===== Time/FPS Measurement =====  */
	struct timeval tic;
	struct timeval toc;
	struct timeval timeout;
	double before = 0.0f;
	float maxFPS = 30.0f;
	float minFrameTime = 1000000.0f / maxFPS;
	int frameCounter = 0;
	float frameCounterAcc = 0.0f;
	double after = 0.0f;    // more accurate time measurements
	double currUSec = 0.0f;
	double currMSec = 0.0f;

	/*  ===== Input Argument =====  */
	int camSource;

	/*  ===== Video Capture =====*/
	cap_cv* cap;
	mat_cv* pInImg;
	int inSDataLength = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS;

	/*  ===== DarkNet =====  */
	network* pNet;
	image inS0;
	image* pInS1;

	/*   ===== Detection =====   */
	int defaultDetectionsCount = 20; // this is speculated, only known at run time, and changes value at each detection (nboxes)
	int nBoxes = 0;
	float* pBBox;
	float* pData0;
	float* pData1;
	int* pTrackerID;
	int* pObjectTyp;
	int itr = 0;

	/*   ===== Classes =====   */
	char** ppNames = NULL;
	int classes = 0;

	/*  ===== Temp Stuff =====  */
	char message[50];
	fd_set readfds;


	/*  ===== INIT OF EVERYTHING ===== */
	/** Calculate the amount of classes by counting the given ppNames */
	/** For COCO DATA we already know it is 80  **/

	ppNames = get_labels(NAMES_FILE); // memory issue NO!
	while (ppNames[classes] != NULL)
	{	
		printf("classes: %d ppNames[%d] %s\n", classes, classes, ppNames[classes]);
		classes++;
	}


	/*  ===== All one time mallocs and lmallocs =====  */
	
	pData0 = (float*)_lmalloc(inSDataLength * sizeof(float), "data_struct_0");
	pData1 = (float*)_lmalloc(inSDataLength * sizeof(float), "data_struct_1");
	pTrackerID = (int*)_lmalloc(defaultDetectionsCount *  sizeof(int), "pTrackerID");
	pObjectTyp = (int*)_lmalloc(defaultDetectionsCount *  sizeof(int), "pObjectTyp");
	pBBox = (float*)_lmalloc(defaultDetectionsCount * 4 * sizeof(float), "pBBox");
	
	/*  ===== some arrays need to be prefilled with zeros =====  */
	for (size_t i = 0; i < 4 * defaultDetectionsCount; ++i) {pBBox[i] = 0.0f;};


	/*  ===== Initialize the network =====  */
	/* THIS NEEDS TO BE EXECUTED ON THE SECOND XAVIER */
#pragma oss task inout(pNet) inout(pInS1) inout(pData1) in(classes) node(1) label("init_network_task")
	{
		pNet = load_network_custom(CFG_FILE, WEIGHT_FILE, 1, 1);
		set_batch_network(pNet, 1);
		fuse_conv_batchnorm(*pNet);
		calculate_binary_weights(*pNet);

		pInS1 = (image*)_malloc(sizeof(image), "pInS1");
		pInS1->w = IMAGE_WIDTH;
		pInS1->h = IMAGE_HEIGHT;
		pInS1->c = CHANNELS;
		pInS1->data = pData1;

		init_trackers(classes);
	}


#pragma oss taskwait

	usleep(1 * 1000000);

	/**  =====  open image source =====  */
	/** If a webcam is directly used, open it with opencv */
	char cap_str[202];

	sprintf(cap_str, "shmsrc socket-path=/dev/shm/camera_small ! video/x-raw, format=BGR, width=%i, height=%i, framerate=30/1 ! queue max-size-buffers=5 leaky=2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true",IMAGE_WIDTH,IMAGE_HEIGHT);

	cap = get_capture_video_stream(cap_str);

	//cap = get_capture_video_stream("../data/test3.mp4");

	/** Check if video capture is opened */
	if (!cap) {
		error("Couldn't connect to webcam.\n");
		abort();
	}

	FD_ZERO(&readfds);

	/** If detection is to fast we need a break */
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;

	/**  =====   Initialize Kalmann and Hungarian filter =====  */
	init_trackers(classes);

	/**  =====   intial iteration =====  */
	/* For the first loop iteration an image is needed in pData1 */
	inS0 = get_image_from_stream_resize(cap, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, &pInImg, 0);

	for (size_t i = 0; i < inSDataLength; ++i)
		pData1[i] = inS0.data[i];

	free_image(inS0);
	release_mat(&pInImg);



	/**  =====  LOOP FOREVER!  =====  */
	while (1)
	{
		size_t jj = 0;
		size_t kk = 0;

		before = get_time_point();
		//gettimeofday(&tic, NULL);
		// check stdin for a new maxFPS amount
		/*(STDIN_FILENO, &readfds);
		if (select(1, &readfds, NULL, NULL, &timeout))
		{
			scanf("%s", message);
			maxFPS = atof(message);
			minFrameTime = 1000000.0 / maxFPS;
		}

		// if no FPS are needed and maxFPS equals 0, wait and start from the beginning
		if (maxFPS == 0)
		{
			usleep(1 * 1000000);
			printf("{\"OBJECT_DET_FPS\": 0.0}\n");
			fflush(stdout);
			continue;
		} */

#pragma oss task in(pNet) in(pInS1) in(classes) in(pData1[0;inSDataLength])  \
        out(nBoxes) out(pBBox[0;4*defaultDetectionsCount]) out(pTrackerID[0;defaultDetectionsCount])\
        out(pObjectTyp[0;defaultDetectionsCount]) label("get_fetch_task") node(1)
		{
			int inSDataLength = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS;
			int nBoxesTask = 0;
			int nBoxesTaskThreshed = 0;
			size_t j = 0;
			size_t k = 0;
			box bbox;
			detection* pDets1;
			float nms = .6f;    // 0.4F
			TrackedObject* pTrackedDets;
			int trackedNBoxes = 0;

			network_predict_image(pNet, *pInS1);
			pDets1 = get_network_boxes(pNet, IMAGE_WIDTH, IMAGE_HEIGHT, THRESH, HIER_THRESH, 0, 1, &nBoxesTask, 0);

			nBoxesTaskThreshed = nBoxesTask;
			if(nBoxesTask > defaultDetectionsCount){
				printf("nBoxes(%d) was to big!!! removing some detection.. please increase defaultDetectionsCount \n",nBoxes);
				do_nms_sort(pDets1, nBoxes, classes, nms);
				nBoxesTaskThreshed = defaultDetectionsCount;
			}
			
			updateTrackers(pDets1, nBoxesTaskThreshed, THRESH, &pTrackedDets, &trackedNBoxes, IMAGE_WIDTH, IMAGE_HEIGHT);

			nBoxes = trackedNBoxes;
			
			for (size_t i = 0; i < trackedNBoxes; ++i)
			{
				bbox = pDets1[i].bbox;
				pBBox[j] = bbox.x;
				pBBox[j + 1] = bbox.y;
				pBBox[j + 2] = bbox.w;
				pBBox[j + 3] = bbox.h;

				pTrackerID[i] = pTrackedDets[i].trackerID;
				pObjectTyp[i] =  pTrackedDets[i].objectTyp;
			}

			if (pTrackedDets != NULL)
				free(pTrackedDets);

			trackedNBoxes = 0;
			
			if (pDets1 != NULL)
				_free_detections(pDets1, nBoxesTask);
		} // end of task

#pragma oss task inout(pData0[0; inSDataLength]) inout(cap) node(0) label("get_image_from_stream_resize_task")
		{
			inS0 = get_image_from_stream_resize(cap, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, &pInImg, 0);
			memcpy(pData0, inS0.data, inSDataLength * sizeof(float));
			free_image(inS0);
			release_mat(&pInImg);
		}


#pragma oss taskwait  


		for (size_t i = 0; i < inSDataLength; ++i)
			pData1[i] = pData0[i];


		fflush(stdout);
		printf("{\"DETECTED_OBJECTS\": [");

		if (nBoxes > 0)
		{
			printf("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}",
				pTrackerID[0], ppNames[pObjectTyp[0]], pBBox[0],
				pBBox[1], pBBox[2], pBBox[3]);
			int i = 1;
			for (i = 1; i < nBoxes; ++i)
			{
				printf(", {\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}",
					pTrackerID[i], ppNames[pObjectTyp[i]], pBBox[i*4 + 0],
					pBBox[i*4 + 1], pBBox[i*4 + 2], pBBox[i*4 + 3]);
			}
		}

		printf("]}\n");
		fflush(stdout);

		after = get_time_point();    // more accurate time measurements
		currUSec = (after - before);
		currMSec = currUSec * 0.001;

		/*if (currUSec < minFrameTime)
		{
			usleep(minFrameTime - currUSec);
			currUSec = minFrameTime;
		} */
		frameCounterAcc += currUSec;
		frameCounter += 1;
		if (frameCounter > maxFPS)
		{
			fflush(stdout);
			//printf("{\"OBJECT_DET_FPS\": %.2f, \"Iteration\": %d, \"maxFPS\": %f, \"lastCurrMSec\": %f}\n", (1000000. / (frameCounterAcc / frameCounter)), itr, maxFPS, currMSec);
			printf("{\"OBJECT_DET_FPS\": %.2f}\n", (1000000. / (frameCounterAcc / frameCounter)));
			fflush(stdout);
			frameCounterAcc = 0.0;
			frameCounter = 0;
		} 

		++itr;
	} // end of while

#pragma oss taskwait

	_lfree(pBBox, 4 * defaultDetectionsCount * sizeof(float));
	_lfree(pData0, inSDataLength * sizeof(float));
	_lfree(pData1, inSDataLength * sizeof(float));
	_lfree(pTrackerID, defaultDetectionsCount * sizeof(int));
	_lfree(pObjectTyp, defaultDetectionsCount * sizeof(int));
	
	// TODO: At one point pInS1, pInS1->data and pNet need to be freed

	return 0;
}


/*
NOTES:

 1- MYsort.c line 37 missing brackets of for loop
 2-

*/
