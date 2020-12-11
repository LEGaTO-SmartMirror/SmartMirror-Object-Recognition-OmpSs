#include "YoloTensorRTWrapper.h"
#include "SORT.h"
#include "Timer.h"
#include "YoloTRT.h"

#include <iostream>

const uint32_t CLASS_COUNT = 80;

const uint32_t WIDTH  = 416;
const uint32_t HEIGHT = 416;

YoloTRT *g_pYolo;
cv::VideoCapture g_cap;

Timer g_timer;

SORT g_sortTrackers[CLASS_COUNT];
std::size_t g_lastCnt;

#define MAX_BUFFERS 2

cv::Mat g_frame[MAX_BUFFERS];
YoloTRT::YoloResults g_yoloResults[MAX_BUFFERS];

extern "C"
{
	struct yoloTRT
	{
	};

	int InitVideoStream(const char *pStr)
	{
		g_cap.open(pStr);
		if (!g_cap.isOpened())
		{
			std::cerr << "Unable to open video stream: " << pStr << std::endl;
			return 0;
		}

		g_timer.Start();

		return 1;
	}

	void InitYoloTensorRT()
	{
		// Set TensorRT log level
		TrtLog::gLogger.setReportableSeverity(TrtLog::Severity::kWARNING);

		g_pYolo = new YoloTRT(true, 0.35f);

		// Initialize SORT tracker for each class
		for (SORT &s : g_sortTrackers)
			s = SORT(5, 3);

		// Set last tracking count to 0
		g_lastCnt = 0;
	}

	void PrintDetections(const TrackingObjects &trackers)
	{
		std::cout << "{\"DETECTED_OBJECTS\": [";

		for (const TrackingObject &t : trackers)
		{
			std::cout << string_format("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}", t.trackingID, t.name.c_str(), t.bBox.x, t.bBox.y, t.bBox.width, t.bBox.height);
			// std::cout << "ID: " << t.trackingID << " - Name: " << t.name << std::endl;
		}

		g_lastCnt = trackers.size();

		std::cout << string_format("], \"DETECTED_OBJECTS_AMOUNT\": %llu }\n", g_lastCnt);
	}

	void ProcessDetections(const uint8_t buffer)
	{
		const YoloTRT::YoloResults &results = g_yoloResults[buffer % MAX_BUFFERS];

		std::map<uint32_t, TrackingObjects> trackingDets;

		for (const YoloTRT::YoloResult &r : results)
		{
			trackingDets.try_emplace(r.ClassID(), TrackingObjects());
			trackingDets[r.ClassID()].push_back({ { r.x, r.y, r.w, r.h }, static_cast<uint32_t>(std::abs(r.Conf() * 100)), r.ClassName() });
		}

		TrackingObjects trackers;
		TrackingObjects dets;

		for (const auto &[classID, tracker] : enumerate(g_sortTrackers))
		{
			if (trackingDets.count(classID))
				dets = trackingDets[classID];
			else
				dets = TrackingObjects();

			TrackingObjects t = tracker.Update(dets);
			trackers.insert(std::end(trackers), std::begin(t), std::end(t));
		}

		// #error This requires a few more checks to catch all possible changes
		if (trackers.size() != g_lastCnt)
			PrintDetections(trackers);
	}

	void GetNextFrame(uint8_t *pData)
	{
		if(!pData) return;
		cv::Mat frame;
		if (g_cap.read(frame))
		{
			std::size_t size = frame.total() * frame.channels();
			cv::Mat flat     = frame.reshape(1, size);
			std::memcpy(pData, flat.ptr(), size);
			// cv::imwrite("out1.jpg", frame);
		}
	}

	void c2CvMat(uint8_t* pData, const int32_t height, const int32_t width, const uint8_t buffer)
	{
		if (pData == nullptr) return;

		g_frame[buffer % MAX_BUFFERS] = cv::Mat(height, width, CV_8UC3, pData);
	}

	void ProcessNextFrame(const uint8_t buffer)
	{
		// cv::Mat frame = cv::Mat(height, width, CV_8UC3, pData);
		// cv::imwrite("out2.jpg", frame);

		if(!g_frame[buffer % MAX_BUFFERS].empty())
			g_yoloResults[buffer % MAX_BUFFERS] = g_pYolo->Infer(g_frame[buffer % MAX_BUFFERS]);
	}

	void Cleanup()
	{
		delete g_pYolo;
	}

	void CheckFPS(uint32_t* pFrameCnt)
	{
		if (g_timer.GetElapsedTimeInMilliSec() >= 1000.0)
		{
			g_timer.Stop();
			std::cout << "Frames: " << (*pFrameCnt) << "| Time: " << g_timer
					  << " | Avg Time: " << g_timer.GetElapsedTimeInMilliSec() / (*pFrameCnt)
					  << " | FPS: " << 1000 / (g_timer.GetElapsedTimeInMilliSec() / (*pFrameCnt)) << std::endl;
			g_timer.Start();
			(*pFrameCnt) = 0;
		}
	}
}
