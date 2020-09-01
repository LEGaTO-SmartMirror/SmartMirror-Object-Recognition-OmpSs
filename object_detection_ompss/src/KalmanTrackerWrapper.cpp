#include "KalmanTracker.h"
#include "KalmanTrackerWrapper.h"

extern "C" {
        KalmanTracker* newMyKalmanTracker(box b) {
				Rect_<float> updatevalue = Rect_<float>(b.x, b.y, b.w, b.h);
                return new KalmanTracker(updatevalue);
        }
      
        box MyKalmanPredict(KalmanTracker* v){
			StateType return_mat = v->predict();
			
			box a;
			a.x = return_mat.x;
			a.y = return_mat.y;
			a.w = return_mat.width;
			a.h = return_mat.height;
			
			return a;
		}
		
		void MyKalmanUpdate(KalmanTracker* v, box b) {
			Rect_<float> updatevalue = Rect_<float>(b.x, b.y, b.w, b.h);
						
			v->update(updatevalue);
		}
		
		size_t MyKalmanTimeSinceUpdate(KalmanTracker* v){
			return v->m_time_since_update;
		}	
		
		box MyKalmanGet_state(KalmanTracker* v){
			StateType return_mat = v->get_state();
			
			box a;
			a.x = return_mat.x;
			a.y = return_mat.y;
			a.w = return_mat.width;
			a.h = return_mat.height;
			
			return a;
		}
		
		int MyKalmanGETm_hits(KalmanTracker* v){
			return v->m_hits;
		}
	
		int MyKalmanGETm_hit_streak(KalmanTracker* v){
			return v->m_hit_streak;
		}
		
		int MyKalmanGETm_age(KalmanTracker* v){
			return v->m_age;
		}
		
		int MyKalmanGETm_id(KalmanTracker* v){
			return v->m_id;
		}

        void deleteMyKalmanTracker(KalmanTracker* v) {
                delete v;
        } 
}
