#include "YoloTensorRTWrapper.h"
#include "SORT.h"
#include "Timer.h"
#include "YoloTRT.h"

#include <chrono>
#include <iostream>
#include <thread>

const uint32_t WIDTH  = 416;
const uint32_t HEIGHT = 416;

const double ONE_SECOND = 1000.0;

YoloTRT* g_pYolo;
cv::VideoCapture g_cap;

double g_elapsedTime;
Timer g_timer;

TrackingObjects g_lastTrackings;

SORT* g_pSortTrackers;

#define MAX_BUFFERS 2

cv::Mat g_frame[MAX_BUFFERS];
YoloTRT::YoloResults g_yoloResults[MAX_BUFFERS];

extern "C"
{
	int InitVideoStream(const char* pStr)
	{
		g_cap.open(pStr);
		if (!g_cap.isOpened())
		{
			std::cerr << "Unable to open video stream: " << pStr << std::endl;
			return 0;
		}

		g_elapsedTime = 0;
		g_timer.Start();

		return 1;
	}

	void InitYoloTensorRT(const char* pOnnxFile, const char* pConfigFile, const char* pEngineFile, const char* pClassFile, const int32_t dlaCore)
	{
		// Set TensorRT log level
		TrtLog::gLogger.setReportableSeverity(TrtLog::Severity::kWARNING);

		g_pYolo = new YoloTRT(std::string(pOnnxFile), std::string(pConfigFile), std::string(pEngineFile), std::string(pClassFile), dlaCore, true, 0.35f);

		g_pSortTrackers = new SORT[YoloTRT::GetClassCount()];

		// Initialize SORT tracker for each class
		for (std::size_t i = 0; i < YoloTRT::GetClassCount(); i++)
			g_pSortTrackers[i] = SORT(5, 3);

		// Set last tracking count to 0
		g_lastTrackings.clear();
	}

	void PrintDetections(const TrackingObjects& trackers)
	{
		std::stringstream str("");
		str << "{\"DETECTED_OBJECTS\": [";

		for (const auto& [i, t] : enumerate(trackers))
		{
			str << string_format("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}", t.trackingID, t.name.c_str(), t.bBox.x, t.bBox.y, t.bBox.width, t.bBox.height);
			// Prevent a trailing ',' for the last element
			if(i + 1 < trackers.size()) str << ", ";
			// std::cout << "ID: " << t.trackingID << " - Name: " << t.name << std::endl;
		}

		g_lastTrackings = trackers;

		str << string_format("], \"DETECTED_OBJECTS_AMOUNT\": %llu }\n", g_lastTrackings.size());
		std::cout << str.str() << std::flush;
	}

	void ProcessDetections(const uint8_t buffer)
	{
		bool changed                        = false;
		const YoloTRT::YoloResults& results = g_yoloResults[buffer % MAX_BUFFERS];

		std::map<uint32_t, TrackingObjects> trackingDets;

		for (const YoloTRT::YoloResult& r : results)
		{
			trackingDets.try_emplace(r.ClassID(), TrackingObjects());
			trackingDets[r.ClassID()].push_back({ { r.x, r.y, r.w, r.h }, static_cast<uint32_t>(std::round(r.Conf() * 100)), r.ClassName() });
		}

		TrackingObjects trackers;
		TrackingObjects dets;

		for (std::size_t i = 0; i < YoloTRT::GetClassCount(); i++)
		{
			if (trackingDets.count(i))
				dets = trackingDets[i];
			else
				dets = TrackingObjects();

			TrackingObjects t = g_pSortTrackers[i].Update(dets);
			trackers.insert(std::end(trackers), std::begin(t), std::end(t));
		}

		if (trackers.size() != g_lastTrackings.size())
			changed = true;
		else
		{
			for (const auto& [idx, obj] : enumerate(trackers))
			{
				if (g_lastTrackings[idx] != obj)
				{
					changed = true;
					break;
				}
			}
		}

		if (changed)
			PrintDetections(trackers);
	}

	void GetNextFrame(uint8_t* pData)
	{
		if (!pData) return;
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

		if (!g_frame[buffer % MAX_BUFFERS].empty())
			g_yoloResults[buffer % MAX_BUFFERS] = g_pYolo->Infer(g_frame[buffer % MAX_BUFFERS]);
	}

	void Cleanup()
	{
		delete g_pYolo;
		delete[] g_pSortTrackers;
	}

	void CheckFPS(uint32_t* pFrameCnt, const uint64_t iteration, const float maxFPS)
	{
		g_timer.Stop();

		double minFrameTime = 1000.0 / maxFPS;
		double itrTime      = g_timer.GetElapsedTimeInMilliSec();
		double fps;
		g_elapsedTime += itrTime;
		fps = 1000 / (g_elapsedTime / (*pFrameCnt));

		if (g_timer.GetElapsedTimeInMilliSec() < minFrameTime)
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int32_t>(std::round(minFrameTime - itrTime))));

		if (g_elapsedTime >= ONE_SECOND)
		{
			if (fps > maxFPS) fps = maxFPS;
			// std::cout << "Frames: " << (*pFrameCnt) << "| Time: " << g_timer
			// 		  << " | Avg Time: " << g_elapsedTime / (*pFrameCnt)
			// 		  << " | FPS: " << 1000 / (g_elapsedTime / (*pFrameCnt)) << std::endl;

			std::cout << string_format("{\"OBJECT_DET_FPS\": %.2f, \"Iteration\": %d, \"maxFPS\": %.2f, \"lastCurrMSec\": %.2f}\n",
									   fps, iteration, maxFPS, itrTime)
					  << std::flush;

			*pFrameCnt    = 0;
			g_elapsedTime = 0;
		}

		g_timer.Start();
	}
}
