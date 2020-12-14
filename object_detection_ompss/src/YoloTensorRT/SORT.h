/* 
 *  File: SORT.h
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
#include <limits>
#include <set>
#include <vector>

#include "HungarianAlgorithm.h"
#include "KalmanBoxTracker.h"
#include "Types.h"
#include "Utils.h"

class SORT
{
	static constexpr double IOU_THRESHOLD = 0.5;

public:
	SORT(const uint32_t& maxAge = 1, const uint32_t& minHits = 3) :
		m_maxAge(maxAge),
		m_minHits(minHits),
		m_trackers(),
		m_frameCount(0)
	{
	}

	TrackingObjects Update(const TrackingObjects& dets)
	{
		m_frameCount++;

		BBox box;
		BBoxes predictedBoxes;

		if (m_trackers.empty() && dets.empty())
			return TrackingObjects();

		for (std::vector<KalmanBoxTracker>::iterator it = m_trackers.begin(); it != m_trackers.end();)
		{
			box = it->Predict();
			if (box.x >= 0 && box.y >= 0)
			{
				predictedBoxes.push_back(box);
				it++;
			}
			else
			{
				it = m_trackers.erase(it);
			}
		}

		// =============================================================================

		std::size_t trkNum  = predictedBoxes.size();
		std::size_t detNum  = dets.size();
		IOUMatrix iouMatrix = IOUMatrix(trkNum, IOUVector(detNum, 0));

		for (std::size_t i = 0; i < trkNum; i++)
		{
			for (std::size_t j = 0; j < detNum; j++)
			{
				iouMatrix[i][j] = 1.0 - getIOU(predictedBoxes[i], dets[j].bBox);
			}
		}

		std::vector<int32_t> assignment;
		if (trkNum > 0)
		{
			HungarianAlgorithm ha;
			ha.Solve(iouMatrix, assignment);
		}

		std::set<int32_t> unmatchedDetections;
		std::set<int32_t> unmatchedTrajectories;
		std::set<int32_t> allItems;
		std::set<int32_t> matchedItems;
		std::vector<cv::Point> matchedPairs;

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (uint32_t n = 0; n < detNum; n++)
				allItems.insert(n);

			for (uint32_t i = 0; i < trkNum; i++)
				matchedItems.insert(assignment[i]);

			std::set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
								std::insert_iterator<std::set<int32_t>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (uint32_t i = 0; i < trkNum; i++)
			{
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
			}
		}

		// filter out matched with low IOU
		for (uint32_t i = 0; i < trkNum; i++)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;

			if (1 - iouMatrix[i][assignment[i]] < IOU_THRESHOLD)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		// =============================================================================

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int32_t detIdx;
		int32_t trkIdx;
		for (const cv::Point& point : matchedPairs)
		{
			trkIdx = point.x;
			detIdx = point.y;
			m_trackers[trkIdx].Update(dets[detIdx].bBox, dets[detIdx].name);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanBoxTracker tracker = KalmanBoxTracker(dets[umd].bBox, dets[umd].name);
			m_trackers.push_back(tracker);
		}

		TrackingObjects frameTrackingResult;

		// get trackers' output
		for (std::vector<KalmanBoxTracker>::iterator it = m_trackers.begin(); it != m_trackers.end();)
		{
			if (it->GetTimeSinceUpdate() < 1 && (it->GetHitStreak() >= m_minHits || m_frameCount <= m_minHits))
				frameTrackingResult.push_back(TrackingObject(it->GetState(), 0, it->GetName(), it->GetID() + 1));

			it++;

			// remove dead tracklet
			if (it != m_trackers.end() && it->GetTimeSinceUpdate() > m_maxAge)
				it = m_trackers.erase(it);
		}

		return frameTrackingResult;
	}

	void ResetCounter() const
	{
		KalmanBoxTracker::ResetCounter();
	}

	bool IsTrackersEmpty() const
	{
		return m_trackers.empty();
	}

	std::size_t GetTrackerCount() const
	{
		return m_trackers.size();
	}

private:
	// Computes IOU between two bounding boxes
	double getIOU(const BBox& bbTest, const BBox& bbGt)
	{
		float in = (bbTest & bbGt).area();
		float un = bbTest.area() + bbGt.area() - in;

		if (un < std::numeric_limits<double>::epsilon())
			return 0;

		double res = static_cast<double>(in / un);
		if (res < 0.0) res = 0.0;
		if (res >= 1.0) res = 1.0;

		return res;
	}

private:
	uint32_t m_maxAge;
	uint32_t m_minHits;
	std::vector<KalmanBoxTracker> m_trackers;
	uint32_t m_frameCount;
};