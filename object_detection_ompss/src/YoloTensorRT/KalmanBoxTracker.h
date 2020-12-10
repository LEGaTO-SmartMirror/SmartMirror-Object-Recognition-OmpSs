/* 
 *  File: KalmanBoxTracker.h
 *  Copyright (c) 2020 Florian Porrmann
 *  
 *  MIT License
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *  
 */

#pragma once

#include <cstdint>
#include <opencv2/video/tracking.hpp>
#include <vector>

#include "Types.h"

class KalmanBoxTracker
{
	static constexpr uint32_t DIM_X = 7;
	static constexpr uint32_t DIM_Z = 4;

public:
	KalmanBoxTracker(const BBox &initRect = BBox(), const std::string &name = "") :
		m_kf(cv::KalmanFilter(DIM_X, DIM_Z, 0)),
		m_measurement(cv::Mat::zeros(DIM_Z, 1, CV_32F)),
		m_history(),
		m_timeSinceUpdate(0),
		m_hits(0),
		m_hitStreak(0),
		m_age(0),
		m_id(s_count++),
		m_name(name)
	{
		m_kf.transitionMatrix = (cv::Mat1f(DIM_X, DIM_X) << 1, 0, 0, 0, 1, 0, 0,
								 0, 1, 0, 0, 0, 1, 0,
								 0, 0, 1, 0, 0, 0, 1,
								 0, 0, 0, 1, 0, 0, 0,
								 0, 0, 0, 0, 1, 0, 0,
								 0, 0, 0, 0, 0, 1, 0,
								 0, 0, 0, 0, 0, 0, 1);

		cv::setIdentity(m_kf.measurementMatrix);
		cv::setIdentity(m_kf.processNoiseCov, cv::Scalar::all(1e-2));
		cv::setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(1e-1));
		cv::setIdentity(m_kf.errorCovPost, cv::Scalar::all(1));

		// initialize state vector with bounding box in [cx,cy,s,r] style
		initBBMatrix(m_kf.statePost, initRect);
	}

	const uint32_t &GetTimeSinceUpdate() const
	{
		return m_timeSinceUpdate;
	}

	const uint32_t &GetHits() const
	{
		return m_hits;
	}

	const uint32_t &GetHitStreak() const
	{
		return m_hitStreak;
	}

	const uint32_t &GetAge() const
	{
		return m_age;
	}

	const uint32_t &GetID() const
	{
		return m_id;
	}

	BBox Predict()
	{
		// predict
		cv::Mat p = m_kf.predict();
		m_age++;

		if (m_timeSinceUpdate > 0)
			m_hitStreak = 0;
		m_timeSinceUpdate++;

		BBox predictBox = getRectXysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

		m_history.push_back(predictBox);
		return predictBox;
	}

	// Update the state vector with observed bounding box.
	void Update(const BBox &bBox, const std::string name)
	{
		m_name = name;

		m_timeSinceUpdate = 0;
		m_history.clear();
		m_hits++;
		m_hitStreak++;

		// Measurement
		initBBMatrix(m_measurement, bBox);

		// Update
		m_kf.correct(m_measurement);
	}

	// Return the current state vector
	BBox GetState() const
	{
		cv::Mat s = m_kf.statePost;
		return getRectXysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
	}

	const std::string &GetName() const
	{
		return m_name;
	}

	static void ResetCounter()
	{
		s_count = 0;
	}

private:
	void initBBMatrix(cv::Mat &mat, const BBox &bBox)
	{
		mat.at<float>(0, 0) = bBox.x + bBox.width / 2.0f;
		mat.at<float>(1, 0) = bBox.y + bBox.height / 2.0f;
		mat.at<float>(2, 0) = bBox.area();
		mat.at<float>(3, 0) = bBox.width / bBox.height;
	}

	BBox getRectXysr(const float &cx, const float &cy, const float &s, const float &r) const
	{
		float w = std::sqrt(s * r);
		float h = s / w;
		float x = (cx - w / 2.0f);
		float y = (cy - h / 2.0f);

		if (x < 0.0f && cx > 0.0f)
			x = 0.0f;
		if (y < 0.0f && cy > 0.0f)
			y = 0.0f;

		return BBox(x, y, w, h);
	}

private:
	cv::KalmanFilter m_kf;
	cv::Mat m_measurement;
	BBoxes m_history;

	uint32_t m_timeSinceUpdate;
	uint32_t m_hits;
	uint32_t m_hitStreak;
	uint32_t m_age;
	uint32_t m_id;

	std::string m_name;

	static uint32_t s_count;
};

uint32_t KalmanBoxTracker::s_count = 0;