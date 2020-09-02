#include "MYSort.h"


void init_trackers(size_t max_index){
	 
	trackers = (struct KalmanTracker***) malloc(sizeof(struct KalmanTracker**) * max_index);
	tracker_types = max_index;
	tracker_amount = (size_t*) malloc(sizeof(size_t) * max_index);
		
	int i;
	for(i=0;i<	max_index ; ++i){
		trackers[i] = NULL;
		tracker_amount[i] = 0;
	}
		
	dets_sorted = (detection***) malloc(sizeof(detection**) *max_index);
	dets_sorted_number = (size_t*) malloc(sizeof(size_t) * max_index);
	for(i=0;i<	max_index ; ++i){
		dets_sorted_number[i] = 0;
		dets_sorted[i] = NULL;
	}
		
	dets_predictions = (box**)  malloc(sizeof(box*) * max_index);
	
}

static void extentTrackers(size_t index, box inital_rect){

	int i = 0;
	
	
	tracker_amount[index] += 1;
	
	struct KalmanTracker** new_trackers =(struct KalmanTracker**) malloc(sizeof(struct KalmanTracker*) * tracker_amount[index]);
	box* new_dets_predictions = (box*) malloc(sizeof(box) * tracker_amount[index]);

	for(i=0 ; i<tracker_amount[index] -1 ; ++i){
		
		if(i < tracker_amount[index]){
			new_trackers[i] = trackers[index][i];
			new_dets_predictions[i] = dets_predictions[index][i];
		}else{

		}
	}

	new_trackers[tracker_amount[index] -1] = newMyKalmanTracker(inital_rect);
	

	if (tracker_amount[index] > 1){
		free(trackers[index]);
		free(dets_predictions[index]);
	}
		
	trackers[index] = new_trackers;		
	dets_predictions[index] = new_dets_predictions;
	
}

static void removeTracker(size_t index, size_t removeIndex){
	
	if (tracker_amount[index] <= removeIndex){
		printf("error: you tried remove a not existing tracker!\n");
		return;
	}
	
	int i = 0;
	
	struct KalmanTracker** new_trackers = NULL;
	box* new_dets_predictions = NULL;
			
	if (tracker_amount[index] > 1) {
	
		new_trackers = (struct KalmanTracker**) malloc(sizeof(struct KalmanTracker*) * (tracker_amount[index] - 1));
		new_dets_predictions = (box*) malloc(sizeof(box) * (tracker_amount[index] - 1));
	
		for(i=0 ; i < tracker_amount[index] ; ++i){
			if (i < removeIndex){
				new_trackers[i] = trackers[index][i];
				new_dets_predictions[i] = dets_predictions[index][i];
			}
			if (i > removeIndex){
				new_trackers[i-1] = trackers[index][i];
				new_dets_predictions[i-1] = dets_predictions[index][i];
			}
		}
	}
	
	deleteMyKalmanTracker(trackers[index][removeIndex]);
	

	if (trackers[index] != NULL){
		free(trackers[index]);
	}
	trackers[index] = new_trackers;
	
	if ( dets_predictions[index] != NULL){
		free(dets_predictions[index]);	
	}
	dets_predictions[index] = new_dets_predictions;
	
	tracker_amount[index] = tracker_amount[index]-1;
	
}

 

void updateTrackers(detection* dets, int nboxes, float thresh, TrackedObject** return_dets, int* return_nboxes, size_t image_width,size_t image_height  ){
	
	size_t i = 0;
	size_t j = 0;
	size_t k = 0;

	*return_dets = NULL;
	*return_nboxes = 0;
	
	//--------------------------------------------------------
	//first we need to sort the detections into types
	for(i=0;i<nboxes; ++i){
		for (j = 0; j < tracker_types; ++j) {
			if (dets[i].prob[j] > thresh) {
				addDetToArray(j,&dets[i]);
			}
		}
	}
	
	//--------------------------------------------------------
	// loop trough all types..
	size_t actual_type = 0;
	size_t trkNum = 0;
	size_t detNum = 0;
	returned_object_amount = 0;
	
	for(actual_type=0; actual_type<tracker_types; ++actual_type){	

		trkNum = tracker_amount[actual_type];
		detNum = dets_sorted_number[actual_type];
			
		// if no tracker or detection is present.. nothing needs to be done		
		if (trkNum == 0 && detNum == 0)
			continue;
		

		//--------------------------------------------------------
		// get predictions of each tracker
		for (i = 0; i <tracker_amount[actual_type]; ++i) {
			dets_predictions[actual_type][i] = MyKalmanPredict(trackers[actual_type][i]);
			// remove nan value
			if(isnan(dets_predictions[actual_type][i].x))
				dets_predictions[actual_type][i].x = 0.0;
			if(isnan(dets_predictions[actual_type][i].y))
				dets_predictions[actual_type][i].y = 0.0;
			if(isnan(dets_predictions[actual_type][i].h))
				dets_predictions[actual_type][i].h = 0.0;
			if(isnan(dets_predictions[actual_type][i].w))
				dets_predictions[actual_type][i].w = 0.0;
				
		}	
		//--------------------------------------------------------
		// combine tracker with a high iou
		// therefore remove the last tracker because it should be younger
		
		for (i = trkNum; i > 1; --i){
			for (j = i-1; j > 0; --j){
				float iou_tmp = calculateIOU(dets_predictions[actual_type][i-1],dets_predictions[actual_type][j-1],image_width, image_height);
				if (iou_tmp > TrackerIOUsimThreshhold){
						removeTracker(actual_type,i-1);
				}
			}
		}
		
		trkNum = tracker_amount[actual_type];
		
		//--------------------------------------------------------
		// assign detections to tracker
	
		//------------------------
		// calculate iou in matrix tracker x detection
	
		float** distMatrix = (float **) malloc(trkNum * sizeof(float *));
		
		for (i = 0; i <trkNum; ++i) {
			distMatrix[i] = (float *) malloc(detNum * sizeof(float));

			
			for (j = 0; j < detNum; ++j) {
				distMatrix[i][j] = 1 - calculateIOU(dets_predictions[actual_type][i],dets_sorted[actual_type][j]->bbox,image_width, image_height);
				
				// if the iou is too low it should not be considered
				if(distMatrix[i][j] > distThreshold)
					distMatrix[i][j] = 1.0;
			}
		}	

		//------------------------
		// Hungerian filter for assignment
				
		int* assignment = (int *) malloc(trkNum * sizeof(int));;
		float cost = Solve(trkNum, detNum, distMatrix, assignment);
				
		//--------------------------------------------------------
		// update Tracker 
		
		for (i=0; i < detNum; ++i){
			int index = valueinarray(i,assignment,trkNum);
			
			if ((index > -1) && distMatrix[index][i] < distThreshold){
				MyKalmanUpdate(trackers[actual_type][index],dets_sorted[actual_type][i]->bbox);
			}
			else {  // there are unmatched detections
				extentTrackers(actual_type, dets_sorted[actual_type][i]->bbox);
			}
		} 
		
		//------------------------
		// free everything allocated..
				
		for (i = 0; i <trkNum; ++i)
			free(distMatrix[i]);
			
		free(distMatrix);
		free(assignment);
				
		//------------------------
		// look for old tracker and remove them

		trkNum = tracker_amount[actual_type];
		
		for (i = trkNum; i > 0; --i){
			size_t timeSinceUpdate = MyKalmanTimeSinceUpdate(trackers[actual_type][i-1]);
			if( timeSinceUpdate > max_age){
				removeTracker(actual_type,i-1);
			}
		}
		
		if (tracker_amount[actual_type] > 0){
			//printf("type = %li\n" ,actual_type);
			for (i = 0; i < tracker_amount[actual_type]; ++i){		
				int m_hint_steak = MyKalmanGETm_hit_streak(trackers[actual_type][i]);
				int age = MyKalmanGETm_age(trackers[actual_type][i]);
				int id = MyKalmanGETm_id(trackers[actual_type][i]);
				if((age > 5) || (m_hint_steak > 3)){
					box b = MyKalmanGet_state(trackers[actual_type][i]);
					TrackedObject new_obj;
					new_obj.bbox.x = b.x;
					new_obj.bbox.y = b.y;
					new_obj.bbox.w = b.w;
					new_obj.bbox.h = b.h;
					new_obj.objectTyp = actual_type;
					new_obj.trackerID = id;
					addDetToReturnArray(new_obj);
					*return_nboxes +=1;
				}
			}
		} 
	
	} // end type loop


	if (*return_nboxes > 0) {
		*return_dets = returned_object;
	} else {
		*return_dets = NULL;
	}	
	
	//printf("-------------------------------------------------------- \n");
	
	// delete pointer to detection
	for (i = 0; i < tracker_types; ++i) {
		if(dets_sorted_number[i] > 0){
			free(dets_sorted[i]);		
			dets_sorted_number[i] = 0;
		}
	}
}

static void addDetToArray(size_t index, detection* det){
	
	int i = 0;
	
	detection** new_detections = (detection**) malloc(sizeof(detection*) * (dets_sorted_number[index] + 1));
			
	if (dets_sorted_number[index] > 0){
		for(i = 0; i < dets_sorted_number[index]; ++i){
			new_detections[i] = dets_sorted[index][i];
		}
		free(dets_sorted[index]);		
	}
	
	new_detections[dets_sorted_number[index]] = det;
	
	dets_sorted[index] = new_detections;
	
	dets_sorted_number[index] += 1;
			
}

static void addDetToReturnArray(TrackedObject det){

	int i = 0;
	TrackedObject* new_detections = (TrackedObject*) malloc(sizeof(TrackedObject) * (returned_object_amount + 1));

	if (returned_object_amount > 0){
		for(i = 0; i < returned_object_amount; ++i){
			new_detections[i] = returned_object[i];
		}
		free(returned_object);		
	}
	
	new_detections[returned_object_amount] = det;

	returned_object = new_detections;
	returned_object_amount += 1;
				
}

static int valueinarray(int val, int arr[], size_t n){
	
    int i;
    for(i = 0; i < n; i++)
    {
        if(arr[i] == val)
            return i;
    }
    return -1;
}

static float calculateIOU(box a, box b,size_t image_width,size_t image_height){
	
	float a_ab_x = a.x * image_width;
	float a_ab_y = a.y * image_height;
	float a_ab_w = a.w * image_width;
	float a_ab_h = a.h * image_height;
		
	float b_ab_x = b.x * image_width;
	float b_ab_y = b.y * image_height;
	float b_ab_w = b.w * image_width;
	float b_ab_h = b.h * image_height;
	
	float overlap_x0 = (a_ab_x-a_ab_w/2) > (b_ab_x-b_ab_w/2) ? (a_ab_x-a_ab_w/2) : (b_ab_x-b_ab_w/2);
	float overlap_y0 = (a_ab_y-a_ab_h/2) > (b_ab_y-b_ab_h/2) ? (a_ab_y-a_ab_h/2) : (b_ab_y-b_ab_h/2);
	
	float overlap_x1 = (a_ab_x+a_ab_w/2) < (b_ab_x+b_ab_w/2) ? (a_ab_x+a_ab_w/2) : (b_ab_x+b_ab_w/2);
	float overlap_y1 = (a_ab_y+a_ab_h/2) < (b_ab_y+b_ab_h/2) ? (a_ab_y+a_ab_h/2) : (b_ab_y+b_ab_h/2);
	
	float w = overlap_x1 - overlap_x0;
	float h = overlap_y1 - overlap_y0;
	
	w = w < 0 ? 0.0 : w;
	h = h < 0 ? 0.0 : h;
	
	float wh = w * h;
	
	float o = wh / ((a_ab_w * a_ab_h) + (b_ab_w * b_ab_h) - wh);
		
	return o;
}
