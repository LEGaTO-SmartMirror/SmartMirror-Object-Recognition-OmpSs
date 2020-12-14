#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include "common.h"
#include "InitTensorRTParser.h"


#include <nanos6/debug.h>
#include <nanos6/cluster.h>

//#define CLASSES         80 // 80 item in coco.ppNames file 
#define THRESH		.6
#define HIER_THRESH	.4
#define CHANNELS	3
#define IMAGE_WIDTH	416
#define IMAGE_HEIGHT	416
#define INSDATALENGTH 	(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS)
#define CFG_FILE	"../data/yolov4.cfg"
#define WEIGHT_FILE	"../data/yolov4.weights"
//#define CFG_FILE	"../data/enet-coco.cfg"
//#define WEIGHT_FILE	"../data/enetb0-coco_final.weights"
#define NAMES_FILE	"../data/coco.names"
#define PRINT_THRESHOLD 0.005

bool CompareFloats2 (float A, float B) 
{
	if((fabs(A - B) < PRINT_THRESHOLD) == 1)
		return false;
	else 
		return true;

}

void printDetections(int nBoxes, float* pBBox, int* pObjectTyp, int* pTrackerID, char** ppNames){
	fflush(stdout);
	printf("{\"DETECTED_OBJECTS\": [");
	if (nBoxes > 0) {
		printf("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}",
			pTrackerID[0], ppNames[pObjectTyp[0]], pBBox[0], pBBox[1], pBBox[2], pBBox[3]);
			for (int i = 1; i < nBoxes; ++i) {
				printf(", {\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}",
					pTrackerID[i], ppNames[pObjectTyp[i]], pBBox[i*4 + 0], pBBox[i*4 + 1], pBBox[i*4 + 2], pBBox[i*4 + 3]);
			}
	}
	printf("], \"DETECTED_OBJECTS_AMOUNT\": %d }\n", nBoxes);
	fflush(stdout);
}


int main(int argc, char** argv)
{

	/* ===== Variable Declarations ===== */
	/*  ===== Time/FPS Measurement =====  */
	double before = 0.0f;
	double after = 0.0f;
	int frameCounter = 0;
	float frameCounterAcc = 0.0f;
	float maxFPS = 30.0f;
	float minFrameTime = (1000000.0 / maxFPS) - 1000;
	int delta = 0.0f;
	double currUSec = 0.0f;
	double currMSec = 0.0f;


	fd_set readfds;
	FD_ZERO(&readfds);
	/** If detection is to fast we need  a break */
	struct timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;
	char message[50];


	/*  ===== default stuff =====  */
	int defaultDetectionsCount = 20; // this is speculated, only known at run time, and changes value at each detection (nboxes)
	int itr = 0;
	float* pData0;
	float* pData1;

	int nBoxes = 0;	
	float* pBBox;
	int* pTrackerID;
	int* pObjectTyp;
	char changed = 0;

	int* nBoxesLastLoop;
	float* pBBoxLastLoop;
	int* pTrackerIDLastLoop;
	int* pObjectTypLastLoop;

	int inSDataLength = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS;
	char** ppNames = NULL;
	int classes = 0;

	/*  ===== Video Capture =====*/
	cap_cv* cap;
	char cap_str[206];
	mat_cv* pInImg;

	/*  ===== DarkNet =====  */
	network* pNet;
	image* pIn0S1;
	image* pIn1S1;
	image inS0;
	detection** pDets0;
	detection** pDets1;
	int* nBoxesTask0;
	int* nBoxesTask1;

	/*  ========= Init starts here =========  */

	if(!InitTensorRTParser())
		printf("Failed to initialize TensorRT Parser\n");

	/**  =====  count how many classes there are! =====  */
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
	for (size_t i = 0; i < inSDataLength; ++i) {
		pData0[i] = 0.0f;
		pData1[i] = 0.0f;
	};

	/*  ===== Initialize everyting for OmpSs =====  */

	#pragma oss task out(nBoxesLastLoop, pTrackerIDLastLoop, pObjectTypLastLoop, pBBoxLastLoop, pDets0, pDets1, nBoxesTask0, nBoxesTask1, pNet, pIn1S1,pIn0S1) \
	in(classes) node(1) label("init_node_1")
	{

		if(!InitTensorRTParser())
			printf("Failed to initialize TensorRT Parser\n");

		printf("{\"STATUS\": \"initilize everything on node 1\"}\n");
		pNet = load_network_custom(CFG_FILE, WEIGHT_FILE, 1, 1);

		set_batch_network(pNet, 1);
		fuse_conv_batchnorm(*pNet);
		calculate_binary_weights(*pNet);

		pIn0S1 = (image*)_malloc(sizeof(image), "pIn0S1");
		pIn0S1->w = IMAGE_WIDTH;
		pIn0S1->h = IMAGE_HEIGHT;
		pIn0S1->c = CHANNELS;
		pIn0S1->data = (float*)_malloc(inSDataLength * sizeof(float), "pIn0S1.data.task");

		pIn1S1 = (image*)_malloc(sizeof(image), "pIn1S1");
		pIn1S1->w = IMAGE_WIDTH;
		pIn1S1->h = IMAGE_HEIGHT;
		pIn1S1->c = CHANNELS;
		pIn1S1->data = (float*)_malloc(inSDataLength * sizeof(float), "pIn1S1.data.task");

		for (size_t i = 0; i < inSDataLength; ++i) {
			pIn0S1->data[i] = 0.0f;
			pIn1S1->data[i] = 0.0f;
		}

		init_trackers(classes);
		pDets0 = (detection**)_malloc(sizeof(detection*), "pointer to pointer of detection 0");
		pDets1 = (detection**)_malloc(sizeof(detection*), "pointer to pointer of detection 1");
		*pDets0 = NULL;
		*pDets1 = NULL;

		nBoxesTask0 = (int*)_malloc(sizeof(int), "box amount of first detection");
		nBoxesTask1 = (int*)_malloc(sizeof(int), "box amount of first detection");
		*nBoxesTask0 = 0;
		*nBoxesTask1 = 0;

		pTrackerIDLastLoop = (int*)_malloc(defaultDetectionsCount *  sizeof(int), "pTrackerID");
		pObjectTypLastLoop = (int*)_malloc(defaultDetectionsCount *  sizeof(int), "pObjectTyp");
		pBBoxLastLoop = (float*)_malloc(defaultDetectionsCount * 4 * sizeof(float), "pBBox");
		for (size_t i = 0; i < 4 * defaultDetectionsCount; ++i) {pBBox[i] = 0.0f; pBBoxLastLoop[i] = 0.0f;};
		for (size_t i = 0; i < defaultDetectionsCount; ++i) {pTrackerID[i] = 0;pTrackerIDLastLoop[i] = 0;};
		for (size_t i = 0; i < defaultDetectionsCount; ++i) {pObjectTyp[i] = 0;pObjectTypLastLoop[i] = 0;};

		nBoxesLastLoop = (int*)_malloc(sizeof(int), "box amount of last detection");
		*nBoxesLastLoop = 0;


	} // end of task

	#pragma oss taskwait

	sleep(5);
	/**  =====  open image source =====  */
	sprintf(cap_str, "shmsrc socket-path=/dev/shm/camera_small ! video/x-raw, format=BGR, width=%i, height=%i, \
		framerate=30/1 ! queue max-size-buffers=5 leaky=2 ! videoconvert ! video/x-raw, format=BGR ! \
		appsink drop=true",IMAGE_WIDTH,IMAGE_HEIGHT);

	cap = get_capture_video_stream(cap_str);


	printf("{\"STATUS\": \"looping starts now\"}\n");
	printf("{\"DETECTED_GESTURES\": []}\n");

	sleep(1);

	/**  =====  LOOP FOREVER!  =====  */
	while (1)
	{
		before = get_time_point();

		// check stdin for a new maxFPS amount
		FD_SET(STDIN_FILENO, &readfds);

		if (select(1, &readfds, NULL, NULL, &timeout)) 	{
			scanf("%s", message);
			maxFPS = atof(message);
			if (maxFPS != 0){
				minFrameTime = (1000000.0 / maxFPS) - 1000;
			}
		}

		// if no FPS are needed and maxFPS equals 0, wait and start from the beginning
		if (maxFPS == 0) {
			sleep(1);
			printf("{\"OBJECT_DET_FPS\": 0.0}\n");
			fflush(stdout);
			continue;
		}

		if(itr % 2 == 0){

			#pragma oss task in(pData1[0; inSDataLength]) node(1) label("even_copy")
			{

				for (size_t i = 0; i < inSDataLength; ++i){
					pIn1S1->data[i] = pData1[i];
				}

				//memcpy(pIn1S1->data, pData1, inSDataLength * sizeof(float));
			} // end task

			#pragma oss task node(1) label("even_predict")
			{
				network_predict_image(pNet, *pIn0S1);
				*pDets0 = get_network_boxes(pNet, IMAGE_WIDTH, IMAGE_HEIGHT, THRESH, HIER_THRESH, 0, 1, nBoxesTask0, 0);
			}

			#pragma oss task out(nBoxes) out(pBBox[0;4*defaultDetectionsCount]) \
			out(pTrackerID[0;defaultDetectionsCount]) out(pObjectTyp[0;defaultDetectionsCount]) \
			out(changed) node(1) label("even_track")
			{

				int inSDataLength = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS;	
				int nBoxesTaskThreshed = 0;
				box bbox;	
				float nms = .6f;    // 0.4F
				TrackedObject* pTrackedDets;
				int trackedNBoxes = 0;
				changed = 0;

				nBoxesTaskThreshed = *nBoxesTask1;
				if(nBoxesTaskThreshed > defaultDetectionsCount){
					printf("nBoxes(%d) was to big!!! removing some detection.. please increase defaultDetectionsCount \n",*nBoxesTask1);
					do_nms_sort(*pDets1, nBoxesTaskThreshed, classes, nms);
					nBoxesTaskThreshed = defaultDetectionsCount;
				}	
				
				updateTrackers(*pDets1, *nBoxesTask1, THRESH, &pTrackedDets, &trackedNBoxes, IMAGE_WIDTH, IMAGE_HEIGHT);
				
								
				if(*nBoxesLastLoop != trackedNBoxes){changed = 1;}

				*nBoxesLastLoop = trackedNBoxes;
				nBoxes = trackedNBoxes;

				for (size_t i = 0; i < trackedNBoxes; ++i)
				{
					bbox = pTrackedDets[i].bbox;

					pBBox[i*4] = bbox.x;
					pBBox[i*4 + 1] = bbox.y;
					pBBox[i*4 + 2] = bbox.w;
					pBBox[i*4 + 3] = bbox.h;
						
					pTrackerID[i] = pTrackedDets[i].trackerID;
					pObjectTyp[i] =  pTrackedDets[i].objectTyp;


					if(pTrackerIDLastLoop[i] != pTrackerID[i])		{changed = 1;}	
					if(pObjectTypLastLoop[i] != pObjectTyp[i])		{changed = 1;}

					if(CompareFloats2(pBBoxLastLoop[i*4]	,bbox.x))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 1],bbox.y))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 2],bbox.w))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 3],bbox.h))	{changed = 1;} 


					pTrackerIDLastLoop[i] 	= pTrackedDets[i].trackerID;
					pObjectTypLastLoop[i] 	= pTrackedDets[i].objectTyp;

					pBBoxLastLoop[i*4] 	= bbox.x;
					pBBoxLastLoop[i*4 + 1] 	= bbox.y;
					pBBoxLastLoop[i*4 + 2] 	= bbox.w;
					pBBoxLastLoop[i*4 + 3] 	= bbox.h;
				}

				if (pTrackedDets != NULL)
					free(pTrackedDets);

				trackedNBoxes = 0; 

				if (*pDets1 != NULL)
					_free_detections(*pDets1, *nBoxesTask1);

			} // end task

			inS0 = get_image_from_stream_resize(cap, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, &pInImg, 1);
			memcpy(pData0, inS0.data, inS0.w* inS0.h * inS0.c * sizeof(float));


		} else { // odd loop!

			#pragma oss task in(pData0[0; inSDataLength]) node(1) label("odd_copy")
			{

				for (size_t i = 0; i < inSDataLength; ++i){
					pIn0S1->data[i] = pData0[i];
				}
				//memcpy(pIn0S1->data, pData0, inSDataLength * sizeof(float));

			} // end task 

			#pragma oss task node(1) label("odd_predict")
			{
				network_predict_image(pNet, *pIn1S1);
				*pDets1 = get_network_boxes(pNet, IMAGE_WIDTH, IMAGE_HEIGHT, THRESH, HIER_THRESH, 0, 1, nBoxesTask1, 0);
			}

			#pragma oss task out(nBoxes) out(pBBox[0;4*defaultDetectionsCount]) \
			out(pTrackerID[0;defaultDetectionsCount]) out(pObjectTyp[0;defaultDetectionsCount]) \
			out(changed) node(1) label("odd_track")
			{
				int inSDataLength = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS;
				int nBoxesTaskThreshed = 0;
				box bbox;

				float nms = .6f;    // 0.4F
				TrackedObject* pTrackedDets;
				int trackedNBoxes = 0;
				changed = 0;

				nBoxesTaskThreshed = *nBoxesTask0;
				if(nBoxesTaskThreshed > defaultDetectionsCount){
					printf("nBoxes(%d) was to big!!! removing some detection.. please increase defaultDetectionsCount \n",*nBoxesTask0);
					do_nms_sort(*pDets0, nBoxesTaskThreshed, classes, nms);
					nBoxesTaskThreshed = defaultDetectionsCount;
				}

				updateTrackers(*pDets0, nBoxesTaskThreshed, THRESH, &pTrackedDets, &trackedNBoxes, IMAGE_WIDTH, IMAGE_HEIGHT);

				if(*nBoxesLastLoop != trackedNBoxes){changed = 1;}

				*nBoxesLastLoop = trackedNBoxes;
				nBoxes = trackedNBoxes;

				for (size_t i = 0; i < trackedNBoxes; ++i)
				{
					bbox = pTrackedDets[i].bbox;

					pBBox[i*4] = bbox.x;
					pBBox[i*4 + 1] = bbox.y;
					pBBox[i*4 + 2] = bbox.w;
					pBBox[i*4 + 3] = bbox.h;
						
					pTrackerID[i] = pTrackedDets[i].trackerID;
					pObjectTyp[i] =  pTrackedDets[i].objectTyp;


					if(pTrackerIDLastLoop[i] != pTrackerID[i])		{changed = 1;}	
					if(pObjectTypLastLoop[i] != pObjectTyp[i])		{changed = 1;}

					if(CompareFloats2(pBBoxLastLoop[i*4]	,bbox.x))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 1],bbox.y))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 2],bbox.w))	{changed = 1;} 
					if(CompareFloats2(pBBoxLastLoop[i*4 + 3],bbox.h))	{changed = 1;} 


					pTrackerIDLastLoop[i] 	= pTrackedDets[i].trackerID;
					pObjectTypLastLoop[i] 	= pTrackedDets[i].objectTyp;

					pBBoxLastLoop[i*4] 	= bbox.x;
					pBBoxLastLoop[i*4 + 1] 	= bbox.y;
					pBBoxLastLoop[i*4 + 2] 	= bbox.w;
					pBBoxLastLoop[i*4 + 3] 	= bbox.h;
				}

				if (pTrackedDets != NULL)
					free(pTrackedDets);

				trackedNBoxes = 0;

				if (*pDets0 != NULL)
					_free_detections(*pDets0, *nBoxesTask0); 
			} // end task

			inS0 = get_image_from_stream_resize(cap, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, &pInImg, 1);
			memcpy(pData1, inS0.data, inS0.w* inS0.h * inS0.c * sizeof(float));

		} // end of even or odd if

		free_image(inS0);
		release_mat(&pInImg);

		#pragma oss taskwait

		if (changed != 0) {
			printDetections(nBoxes, pBBox, pObjectTyp, pTrackerID, ppNames);
		}

		if (itr > 2000000000)
			itr = 0;
		else
			itr += 1;

		after = get_time_point();    // more accurate time measurements
		currUSec = (after - before);
		
		if (currUSec < minFrameTime){
			usleep(minFrameTime - currUSec);
			currUSec = minFrameTime;
		}
		
		currMSec = currUSec * 0.001;

		frameCounterAcc += currUSec;
		frameCounter += 1;
		if (frameCounter > maxFPS)
		{
			fflush(stdout);
			printf("{\"OBJECT_DET_FPS\": %.2f, \"Iteration\": %d, \"maxFPS\": %.2f, \"lastCurrMSec\": %.2f}\n",
			 (1000000. / (frameCounterAcc / frameCounter)), itr, maxFPS, currMSec);
			fflush(stdout);
			frameCounterAcc = 0.0;
			frameCounter = 0;
		}

	} // end of while

	_lfree(pData0, inSDataLength * sizeof(float));
	_lfree(pData1, inSDataLength * sizeof(float));


	return 0;
}

