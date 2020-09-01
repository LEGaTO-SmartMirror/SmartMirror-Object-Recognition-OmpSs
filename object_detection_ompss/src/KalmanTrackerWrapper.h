#ifndef __KALMANTRACKERWRAPPER_H
#define __KALMANTRACKERWRAPPER_H 2

#include <stdlib.h>
#include <darknet.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KalmanTracker KalmanTracker;

KalmanTracker* newMyKalmanTracker(box b);

//void MyClass_int_set(KalmanTracker* v, int i);

//int MyClass_int_get(KalmanTracker* v);

box MyKalmanPredict(KalmanTracker* v);
void MyKalmanUpdate(KalmanTracker* v, box b);
size_t MyKalmanTimeSinceUpdate(KalmanTracker* v);
box MyKalmanGet_state(KalmanTracker* v);
int MyKalmanGETm_hits(KalmanTracker* v);
int MyKalmanGETm_hit_streak(KalmanTracker* v);
int MyKalmanGETm_age(KalmanTracker* v);
int MyKalmanGETm_id(KalmanTracker* v);


void deleteMyKalmanTracker(KalmanTracker* v);

 

#ifdef __cplusplus
}
#endif
#endif
