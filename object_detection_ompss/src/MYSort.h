#ifndef MYSORT_H
#define MYSORT_H

#include "KalmanTrackerWrapper.h"
#include "Hungarian.h"
#include <stdio.h>
#include <darknet.h>


//struct KalmanTracker* tracker;
static struct KalmanTracker*** trackers;
static size_t tracker_types;
static size_t* tracker_amount;

#define max_age                     5
#define distThreshold               0.95
#define TrackerIOUsimThreshhold     0.8

// for an efficent tracking all detecitons needs to be sorted for object types.
// detection[tracker_types][dets_sorted_number[tracker_types]]
static detection*** dets_sorted;
static size_t* dets_sorted_number;

typedef struct TrackedObject{
    box bbox;
    int objectTyp;
    size_t trackerID;
} TrackedObject;

static TrackedObject* returned_object;
static size_t returned_object_amount;

// get the prediction of each tracker of each type to this.
// detection[tracker_types][tracker_amount[tracker_types]]
static box** dets_predictions;

// -----------------------
// Fuctions
// -----------------------

// initialize pointer with the max amount of types
void init_trackers(size_t max_index);

// public 
void updateTrackers(detection* dets, int nboxes, float thresh, TrackedObject** return_dets, int* return_nboxes, size_t _image_width, size_t _image_height);

//internal functions for compuational splitting
static void addDetToArray(size_t index, detection* det);
static void addDetToReturnArray(TrackedObject det);
static void extentTrackers(size_t index, box inital_rect);
static void removeTracker(size_t index, size_t removeIndex);
static int valueinarray(int val, int arr[], size_t n);
static float calculateIOU(box a, box b, size_t _image_width, size_t _image_height);


#endif
