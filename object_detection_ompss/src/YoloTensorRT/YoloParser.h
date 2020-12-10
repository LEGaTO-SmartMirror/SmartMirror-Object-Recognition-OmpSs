/* 
 *  File: YoloParser.h
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

#include "Utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

DEFINE_EXCEPTION(YoloParserException);

enum class YoloType
{
	NON     = 0,
	TINY    = 1 << 0,
	YOLO_V3 = 1 << 1,
	YOLO_V4 = 1 << 2
};

inline YoloType operator|(YoloType a, YoloType b)
{
	return static_cast<YoloType>(static_cast<int>(a) | static_cast<int>(b));
}

inline YoloType operator&(YoloType a, YoloType b)
{
	return static_cast<YoloType>(static_cast<int>(a) & static_cast<int>(b));
}

inline YoloType& operator|=(YoloType& a, YoloType b)
{
	return a = a | b;
	;
}

inline YoloType& operator|=(YoloType& a, int b)
{
	return a = a | static_cast<YoloType>(b);
}

class YoloParser
{
	inline static const std::string YOLO_TAG    = "[yolo]";
	inline static const std::string MASK_TAG    = "mask";
	inline static const std::string ANCHORS_TAG = "anchors";
	inline static const std::string SCALE_TAG   = "scale_x_y";

	static constexpr auto STRING_TO_INT   = [](const std::string& s) { return std::stoi(s); };
	static constexpr auto STRING_TO_FLOAT = [](const std::string& s) { return std::stof(s); };

public:
	using AnchorType = float;
	using Anchors    = std::vector<AnchorType>;
	struct YoloLayer
	{
		uint32_t id;
		Anchors anchors;
		float scale;
		int32_t gridW;
		int32_t gridH;

		int32_t GetGridArea() const
		{
			return gridW * gridH;
		}
	};

	using YoloLayers = std::vector<YoloLayer>;

public:
	YoloParser(const std::string& cfgName = "", const YoloType& type = YoloType::NON) :
		m_cfg(""),
		m_yoloLayers(),
		m_width(-1),
		m_height(-1),
		m_classes(-1),
		m_type(type)
	{
		if (!cfgName.empty())
			Parse(cfgName);
	}

	bool Parse(const std::string& cfgName)
	{
		if (!readConfigFile(cfgName))
		{
			std::cerr << "Failed to parse Yolo config" << std::endl;
			return false;
		}

		readYoloLayers();
		m_width   = readElement("width");
		m_height  = readElement("height");
		m_classes = readElement("classes");

		if (m_type == YoloType::NON)
		{
			// If the config did not contain the scale_x_y property
			// it is probably a Yolo v3
			if (m_yoloLayers.front().scale == 1.0f)
				m_type = YoloType::YOLO_V3;
			else
				m_type = YoloType::YOLO_V4;

			// A tiny yolo only has two anchors
			if (m_yoloLayers.size() == 2)
				m_type |= YoloType::TINY;
		}

		uint32_t factor = 32;

		// For Yolo v4 the order is reverse, i.e., 8 -> 16 -> 32
		// This however only applies to the full Yolo v4 not v4 tiny
		if (m_type == YoloType::YOLO_V4) factor = 8;

		for (YoloLayer& a : m_yoloLayers)
		{
			a.gridW = static_cast<int32_t>(m_width / factor);
			a.gridH = static_cast<int32_t>(m_height / factor);

			if (m_type == YoloType::YOLO_V4)
				factor *= 2;
			else
				factor /= 2;
		}

		return true;
	}

	const int32_t& GetWidth() const
	{
		return m_width;
	}

	const int32_t& GetHeight() const
	{
		return m_height;
	}

	const int32_t& GetClasses() const
	{
		return m_classes;
	}

	const YoloLayers& GetYoloLayers() const
	{
		return m_yoloLayers;
	}

	const YoloLayer& GetYoloLayer(const uint32_t idx) const
	{
		return m_yoloLayers.at(idx);
	}

	const std::size_t GetNumAnchors() const
	{
		if (m_yoloLayers.empty())
			throw(YoloParserException("[GetNumAnchors]: No YoloLayer found - Parsing the config might have failed"));

		return m_yoloLayers.front().anchors.size() / 2;
	}

	const YoloType& GetType() const
	{
		return m_type;
	}

	friend std::ostream& operator<<(std::ostream& os, const YoloParser& yp)
	{
		os << "Width=" << yp.m_width << std::endl
		   << "Height=" << yp.m_height << std::endl
		   << "Classes=" << yp.m_classes;

		for (const YoloLayer& a : yp.m_yoloLayers)
		{
			os << std::endl
			   << "Yolo-Layer [" << a.id << "]:" << std::endl
			   << "Anchor-Values=" << std::flush;

			for (const AnchorType& v : a.anchors)
				os << v << " " << std::flush;

			os << std::endl
			   << "Scale=" << a.scale << std::endl
			   << "Grid-Width=" << a.gridW << std::endl
			   << "Grid-Height=" << a.gridH;
		}

		return os;
	}

private:
	bool readConfigFile(const std::string& cfgName)
	{
		std::ifstream cfgFile(cfgName);

		if (!cfgFile.is_open())
		{
			std::cerr << "Failed to open Yolo config file: \"" << cfgName << "\"" << std::endl;
			return false;
		}

		cfgFile.seekg(0, std::ios::end);
		m_cfg.reserve(cfgFile.tellg());
		cfgFile.seekg(0, std::ios::beg);

		m_cfg.assign((std::istreambuf_iterator<char>(cfgFile)), std::istreambuf_iterator<char>());
		return !m_cfg.empty();
	}

	template<typename T, typename F>
	std::vector<T> readList(const std::string& tag, const std::size_t& startPos, F f) const
	{
		// Find the starting position of the search tag
		std::size_t pos = m_cfg.find(tag, startPos);
		// Return an empty vector if the tag was not found
		if (pos == std::string::npos) return std::vector<T>();

		// Reverse search for the first newline before the tag
		std::size_t nlPos = m_cfg.rfind("\n", pos);
		// Check if a comment sign '#' is present between the newline and the
		// start of the tag, this would mean that this line is commented out
		if (m_cfg.substr(nlPos, pos - nlPos).find("#") != std::string::npos)
		{
			// In that case continue searching for the tag after the current position
			return readList<T>(tag, pos + tag.length(), f);
		}

		// Get the position right after the equal sign
		pos = m_cfg.find("=", pos) + 1;
		// Create a substring containing only the list
		std::string listStr = m_cfg.substr(pos, m_cfg.find("\n", pos) - pos);
		// Remove all spaces in the string version of the list
		listStr.erase(std::remove_if(std::begin(listStr), std::end(listStr), [](const char& c) { return c == ' '; }), std::end(listStr));
		// Split the string list into a integer list
		std::vector<T> list = splitStringT<T>(listStr, ',', f);

		return list;
	}

	std::vector<int32_t> readList(const std::string& tag, const std::size_t& startPos = 0) const
	{
		return readList<int32_t>(tag, startPos, STRING_TO_INT);
	}

	template<typename T, typename F>
	T readElement(const std::string& tag, const T& defaultValue, const std::size_t& startPos, F f) const
	{
		std::vector<T> l = readList<T>(tag, startPos, f);
		if (l.empty()) return defaultValue;

		return l.front();
	}

	int32_t readElement(const std::string& tag, const int32_t& defaultValue = -1, const std::size_t& startPos = 0) const
	{
		return readElement<int32_t>(tag, defaultValue, startPos, STRING_TO_INT);
	}

	void readYoloLayers()
	{
		std::size_t pos = 0;
		m_yoloLayers.clear();

		while ((pos = m_cfg.find(YOLO_TAG, pos)) != std::string::npos)
		{
			YoloLayer a;
			a.id = m_yoloLayers.size();
			pos += YOLO_TAG.length();
			std::vector<int32_t> mask = readList(MASK_TAG, pos);
			Anchors anchors           = readList<float>(ANCHORS_TAG, pos, STRING_TO_FLOAT);
			a.scale                   = readElement<float>(SCALE_TAG, 1.0f, pos, STRING_TO_FLOAT);

			for (const int32_t& m : mask)
			{
				a.anchors.push_back(anchors[m * 2]);
				a.anchors.push_back(anchors[m * 2 + 1]);
			}

			m_yoloLayers.push_back(a);
		}
	}

private:
	std::string m_cfg;
	YoloLayers m_yoloLayers;
	int32_t m_width;
	int32_t m_height;
	int32_t m_classes;
	YoloType m_type;
};
