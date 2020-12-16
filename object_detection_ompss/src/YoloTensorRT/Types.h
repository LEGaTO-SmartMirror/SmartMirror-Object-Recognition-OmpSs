/* 
 *  File: Types.h
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

#include <opencv2/core/core.hpp>
#include <tuple>
#include <vector>

using BBox   = cv::Rect2f;
using BBoxes = std::vector<BBox>;

#if !defined(THRESHOLDED_DIFF)
#ifdef DIFF_THRESHOLD // In case THRESHOLDED_DIFF is not defined but DIFF_THRESHOLD is defined, define THRESHOLDED_DIFF
#define THRESHOLDED_DIFF
#else
#define STRICT_DIFF // Else define STRICT_DIFF
#endif
#endif

#if defined(THRESHOLDED_DIFF) && defined(STRICT_DIFF)
#warning "Both THRESHOLDED_DIFF and STRICT_DIFF are defined, using THRESHOLDED_DIFF"
#undef STRICT_DIFF
#endif

#ifndef DIFF_THRESHOLD
#define DIFF_THRESHOLD 0.005
#endif

#define THRESHED_CMP(a, b, OP) (std::fabs(a - b) OP DIFF_THRESHOLD)

struct TrackingObject
{
	TrackingObject() :
		bBox(),
		score(0),
		trackingID(0),
		name("Unknown"),
		lastCheck(0),
		lastUpdate(0),
		face()
	{
	}

	TrackingObject(const BBox& box, const uint32_t& s, const uint32_t& tID = 0) :
		bBox(box),
		score(s),
		trackingID(tID),
		name("Unknown"),
		lastCheck(-1),
		lastUpdate(0),
		face()
	{
	}

	TrackingObject(const BBox& box, const uint32_t& s, const std::string& nName, const uint32_t& tID = 0) :
		bBox(box),
		score(s),
		trackingID(tID),
		name(nName),
		lastCheck(-1),
		lastUpdate(0),
		face()
	{
	}

	bool operator!=(const TrackingObject& rhs)
	{
		if (this->trackingID != rhs.trackingID) return true;
		if (this->name != rhs.name) return true;
#ifdef THRESHOLDED_DIFF
		if (THRESHED_CMP(this->bBox.x, rhs.bBox.x, >=)) return true;
		if (THRESHED_CMP(this->bBox.y, rhs.bBox.y, >=)) return true;
		if (THRESHED_CMP(this->bBox.width, rhs.bBox.width, >=)) return true;
		if (THRESHED_CMP(this->bBox.height, rhs.bBox.height, >=)) return true;
#endif
#ifdef STRICT_DIFF
		return this->bBox != rhs.bBox;
#endif

		return false;
	}

	bool Valid() const
	{
		return lastUpdate < 5;
	}

	BBox bBox;
	uint32_t score;
	uint32_t trackingID;
	std::string name;
	int32_t lastCheck;
	uint32_t lastUpdate;
	cv::Mat face;
};

using TrackingObjects = std::vector<TrackingObject>;

using IOUType   = double;
using IOUVector = std::vector<IOUType>;
using IOUMatrix = std::vector<IOUVector>;