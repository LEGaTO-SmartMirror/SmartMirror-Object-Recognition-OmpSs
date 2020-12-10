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
uint32_t g_frameCnt;

SORT g_sortTrackers[CLASS_COUNT];
std::size_t g_lastCnt;

YoloTRT::YoloResults g_yoloResults[2];

extern "C"
{
	int InitVideoStream(const char *pStr)
	{
		g_cap.open(pStr);
		if (!g_cap.isOpened())
		{
			std::cerr << "Unable to open video stream: " << pStr << std::endl;
			return 0;
		}

		// Set last frame count to 0
		g_frameCnt = 0;
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

	void ProcessDetections()
	{
		const YoloTRT::YoloResults &results = g_yoloResults[g_frameCnt % 2];
		// if (results.empty() && g_trackerSort.IsTrackersEmpty()) return;

		std::map<uint32_t, TrackingObjects> trackingDets;

		for (const YoloTRT::YoloResult &r : results)
		{
			trackingDets.try_emplace(r.ClassID(), TrackingObjects());
			trackingDets[r.ClassID()].push_back({ { r.x, r.y, r.w, r.h }, static_cast<uint32_t>(std::abs(r.Conf() * 100)), r.ClassName() });
			// std::cout << r << std::endl;
			// cv::rectangle(imgLocal, cv::Point(r.x, r.y), cv::Point(r.x + r.w, r.y + r.h), cv::Scalar(0, 255, 0));
			// cv::putText(imgLocal, string_format("%s - %f", r.ClassName().c_str(), r.Conf()), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(255, 50, 50), 1);
		}

		TrackingObjects trackers;

		// #error This requires a few more checks to catch all possible changes
		for (const auto &[classID, dets] : trackingDets)
		{
			TrackingObjects t = g_sortTrackers[classID].Update(dets);
			trackers.insert(std::end(trackers), std::begin(t), std::end(t));
		}

		if (trackers.size() != g_lastCnt)
		{
			PrintDetections(trackers);
		}
	}

	uint8_t *GetNextFrame()
	{
		uint8_t *pData = nullptr;
		cv::Mat frame;
		if (g_cap.read(frame))
		{
			std::size_t size = frame.total() * frame.channels();
			pData            = new uint8_t[size];
			cv::Mat flat     = frame.reshape(1, size);
			std::memcpy(pData, flat.ptr(), size);
		}

		return pData;
	}

	void ProcessNextFrame(uint8_t *pData, const int32_t height, const int32_t width)
	{
		if (pData == nullptr) return;

		cv::Mat frame = cv::Mat(height, width, CV_8UC3, pData);
		delete[] pData;

		g_yoloResults[g_frameCnt % 2] = g_pYolo->Infer(frame);

		g_frameCnt++;
	}

	void Cleanup()
	{
		delete g_pYolo;
	}

	void CheckFPS()
	{
		if (g_timer.GetElapsedTimeInMilliSec() >= 1000.0)
		{
			g_timer.Stop();
			std::cout << "Frames: " << g_frameCnt << "| Time: " << g_timer
					  << " | Avg Time: " << g_timer.GetElapsedTimeInMilliSec() / g_frameCnt
					  << " | FPS: " << 1000 / (g_timer.GetElapsedTimeInMilliSec() / g_frameCnt) << std::endl;
			g_timer.Start();
			g_frameCnt = 0;
		}
	}
}
