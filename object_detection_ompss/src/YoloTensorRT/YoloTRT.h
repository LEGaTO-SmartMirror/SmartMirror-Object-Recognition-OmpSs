/* 
 *  File: YoloTRT.h
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

#include "Timer.h"
#include "Utils.h"
#include "YoloParser.h"

// == TensorRT includes ==
#include "buffers.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include <NvInfer.h>
// == TensorRT includes ==

// == Yolo Plugin include ==
#include "YoloLayerPlugin.h"
// == Yolo Plugin include ==

// == OpenCV includes ==
#include <opencv2/img_hash/phash.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
// == OpenCV includes ==

#include <algorithm>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::experimental::filesystem;

DEFINE_EXCEPTION(YoloTRTException);

class YoloTRT
{
	inline static const std::vector<std::string> COCO_CLASSES = {
		"person",
		"bicycle",
		"car",
		"motorbike",
		"aeroplane",
		"bus",
		"train",
		"truck",
		"boat",
		"traffic light",
		"fire hydrant",
		"stop sign",
		"parking meter",
		"bench",
		"bird",
		"cat",
		"dog",
		"horse",
		"sheep",
		"cow",
		"elephant",
		"bear",
		"zebra",
		"giraffe",
		"backpack",
		"umbrella",
		"handbag",
		"tie",
		"suitcase",
		"frisbee",
		"skis",
		"snowboard",
		"sports ball",
		"kite",
		"baseball bat",
		"baseball glove",
		"skateboard",
		"surfboard",
		"tennis racket",
		"bottle",
		"wine glass",
		"cup",
		"fork",
		"knife",
		"spoon",
		"bowl",
		"banana",
		"apple",
		"sandwich",
		"orange",
		"broccoli",
		"carrot",
		"hot dog",
		"pizza",
		"donut",
		"cake",
		"chair",
		"sofa",
		"pottedplant",
		"bed",
		"diningtable",
		"toilet",
		"tvmonitor",
		"laptop",
		"mouse",
		"remote",
		"keyboard",
		"cell phone",
		"microwave",
		"oven",
		"toaster",
		"sink",
		"refrigerator",
		"book",
		"clock",
		"vase",
		"scissors",
		"teddy bear",
		"hair drier",
		"toothbrush",
	};

	inline static const std::string ONNX_FILE   = "../data/yolov4-416.onnx";
	inline static const std::string CONFIG_FILE = "../data/yolov4-416.cfg";
	inline static const std::string ENGINE_FILE = "../data/yolov4-416.engine";
	inline static const std::string INPUT_LAYER = "000_net";

	inline static const bool USE_FP16    = true;
	inline static const int32_t DLA_CORE = 0; // -1 => Do not use DLAs

	inline static const bool FORCE_REBUILD = false;

	template<typename T>
	using InferUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	struct YoloResult
	{
		float x;
		float y;
		float w;
		float h;
		float boxConfidence;
		float classID;
		float classProb;

		float Conf() const
		{
			return boxConfidence * classProb;
		}

		void Scale(const uint32_t &width, const uint32_t &height)
		{
			x *= width;
			y *= height;
			w *= width;
			h *= height;
		}

		std::string ClassName() const
		{
			int32_t id = static_cast<int32_t>(classID);
			if (COCO_CLASSES.size() < id)
				return string_format("INVALID_CLASSID: %s", id);
			return COCO_CLASSES.at(id);
		}

		uint32_t ClassID() const
		{
			return static_cast<uint32_t>(classID);
		}

		friend std::ostream &operator<<(std::ostream &os, const YoloResult &dt)
		{
			os << string_format("x=%f - y=%f - w=%f - h=%f - boxC=%f - cID=%d - cProb=%f", dt.x, dt.y, dt.w, dt.h, dt.boxConfidence, static_cast<int32_t>(dt.classID), dt.classProb);
			// os << string_format("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.5f,%.5f], \"w_h\": [%.5f,%.5f]}", 0, dt.ClassName().c_str(), dt.x, dt.y, dt.w, dt.h);
			return os;
		}
	};

	using YoloResults = std::vector<YoloResult>;

public:
	YoloTRT(const bool &autoLoad = false, const float &threshold = 0.3f) :
		m_engine(nullptr),
		m_context(nullptr),
		m_buffers(nullptr),
		m_outputLayer(),
		m_verbose(false),
		m_imgSize(0, 0),
		m_yoloParser(CONFIG_FILE),
		m_threshold(threshold)
	{
		if (autoLoad)
			buildOrLoadEngine(FORCE_REBUILD);
	}

	void InitEngine()
	{
		buildOrLoadEngine(FORCE_REBUILD);
	}

	void SetVerbose(const bool &en = true)
	{
		m_verbose = en;
	}

	void SetThreshold(const float &threshold)
	{
		m_threshold = threshold;
	}

	YoloResults Infer(const cv::Mat &img)
	{
		m_imgSize = img.size();

		Timer timer;

		// ==== Inference ====
		timer.Start();

		processInput(img, false);

		// Copy data from host input buffers to device input buffers
		m_buffers->copyInputToDevice();

		// Execute the inference work
		if (!m_context->executeV2(m_buffers->getDeviceBindings().data()))
			return YoloResults();

		// Copy data from device output buffers to host output buffers
		m_buffers->copyOutputToHost();

		// ==== Inference ====

		YoloResults results = processOutput(img);

		// std::cout << "Inference-Timer: " << timer << std::endl;

		return results;
	}

	static std::size_t GetClassCount()
	{
		return COCO_CLASSES.size();
	}

private:
	void buildOrLoadEngine(const bool &forceRebuild = false)
	{
		if (!forceRebuild && fileExists(ENGINE_FILE))
			loadEngine();
		else
			buildEngine();

		// Create RAII buffer manager object
		m_buffers = std::make_shared<samplesCommon::BufferManager>(m_engine);
	}

	InferUniquePtr<nvinfer1::IBuilderConfig> config;

	void loadEngine()
	{
		// auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLog::gLogger.getTRTLogger()));
		// if (!builder)
		// 	throw(YoloTRTException("Failed to create Builder"));

		// const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

		// auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		// if (!network)
		// 	throw(YoloTRTException("Failed to create Network"));

		// config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		// if (!config)
		// 	throw(YoloTRTException("Failed to create BuilderConfig"));

		// auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, TrtLog::gLogger.getTRTLogger()));
		// if (!parser)
		// 	throw(YoloTRTException("Failed to create Parser"));

		// if (!parser->parseFromFile(ONNX_FILE.c_str(), static_cast<int>(TrtLog::Severity::kERROR)))
		// 	throw(YoloTRTException("Failed to parse ONNX File"));

		std::cout << "Loading serialized engine ... " << std::flush;

		std::ifstream file(ENGINE_FILE, std::ios::binary);
		if (!file)
			throw(YoloTRTException(string_format("[LoadEngine] Failed to open Engine File: %s", ENGINE_FILE)));

		file.seekg(0, file.end);
		std::size_t size = file.tellg();
		file.seekg(0, file.beg);

		std::vector<char> engineData(size);
		file.read(engineData.data(), size);
		file.close();

		InferUniquePtr<nvinfer1::IRuntime> pRuntime{ nvinfer1::createInferRuntime(TrtLog::gLogger.getTRTLogger()) };
		if (!pRuntime)
			throw(YoloTRTException("Failed to create InferRuntime"));

		if (DLA_CORE >= 0)
		{
			std::cout << " - Enabling DLACore=" << DLA_CORE << " - " << std::flush;
			pRuntime->setDLACore(DLA_CORE);
		}

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(pRuntime->deserializeCudaEngine(engineData.data(), size, nullptr), samplesCommon::InferDeleter());
		if (!m_engine)
			throw(YoloTRTException("Failed to deserialize Engine"));

		uint32_t cnt = 0;
		for (int32_t i = 0; i < m_engine->getNbBindings(); i++)
		{
			std::string name = std::string(m_engine->getBindingName(i));
			if (name != INPUT_LAYER)
			{
				m_outputLayer[name] = m_yoloParser.GetYoloLayer(cnt).GetGridArea();
				cnt++;
			}
		}

		m_context = InferUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
		if (!m_context)
			throw(YoloTRTException("Failed to create Execution Context"));

		std::cout << "Done" << std::endl;
	}

	void buildEngine()
	{
		auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLog::gLogger.getTRTLogger()));
		if (!builder)
			throw(YoloTRTException("Failed to create Builder"));

		const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

		auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		if (!network)
			throw(YoloTRTException("Failed to create Network"));

		auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!config)
			throw(YoloTRTException("Failed to create BuilderConfig"));

		auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, TrtLog::gLogger.getTRTLogger()));
		if (!parser)
			throw(YoloTRTException("Failed to create Parser"));

		if (!parser->parseFromFile(ONNX_FILE.c_str(), static_cast<int>(TrtLog::Severity::kERROR)))
			throw(YoloTRTException("Failed to parse ONNX File"));

		// ==== Add Yolo Layer ====
		auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");

		int32_t inputWidth   = m_yoloParser.GetWidth();
		int32_t intputHeight = m_yoloParser.GetHeight();

		int32_t numClasses = m_yoloParser.GetClasses();
		int32_t numAnchors = m_yoloParser.GetNumAnchors();

		std::cout << "Input: " << network->getInput(0)->getName() << std::endl;

		std::vector<nvinfer1::ITensor *> oldTensors;
		std::vector<nvinfer1::ITensor *> newTensors;

		for (int32_t i = 0; i < network->getNbOutputs(); i++)
			oldTensors.push_back(network->getOutput(i));

		if (oldTensors.size() != m_yoloParser.GetYoloLayers().size())
			throw(YoloTRTException(string_format("Output Layer count (%llu) does not match layer count from the config (%llu)", oldTensors.size(), m_yoloParser.GetYoloLayers().size())));

		for (std::size_t i = 0; i < oldTensors.size(); i++)
		{
			const YoloParser::YoloLayer &yLayer = m_yoloParser.GetYoloLayer(i);
			std::cout << "Processing Output Layer: " << oldTensors.at(i)->getName() << std::endl;

			std::vector<nvinfer1::PluginField> pluginAttributes;
			pluginAttributes.push_back(nvinfer1::PluginField("yoloWidth", &yLayer.gridW, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("yoloHeight", &yLayer.gridH, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("inputWidth", &inputWidth, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("inputHeight", &intputHeight, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("numClasses", &numClasses, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("numAnchors", &numAnchors, nvinfer1::PluginFieldType::kINT32, 1));
			pluginAttributes.push_back(nvinfer1::PluginField("anchors", yLayer.anchors.data(), nvinfer1::PluginFieldType::kFLOAT32, yLayer.anchors.size()));
			pluginAttributes.push_back(nvinfer1::PluginField("scaleXY", &yLayer.scale, nvinfer1::PluginFieldType::kFLOAT32, 1));

			nvinfer1::PluginFieldCollection pluginData;
			pluginData.nbFields = pluginAttributes.size();
			pluginData.fields   = pluginAttributes.data();

			nvinfer1::IPluginV2 *pluginObj = creator->createPlugin("YoloLayer_TRT", &pluginData);

			newTensors.push_back(network->addPluginV2(&oldTensors.at(i), 1, *pluginObj)->getOutput(0));
			m_outputLayer[newTensors.back()->getName()] = yLayer.GetGridArea();
		}

		std::cout << "Marking new Outputs ... " << std::flush;
		for (nvinfer1::ITensor *newTensor : newTensors)
			network->markOutput(*newTensor);
		std::cout << "Done" << std::endl;

		std::cout << "Unmarking old Outputs ... " << std::flush;
		for (nvinfer1::ITensor *oldTensor : oldTensors)
			network->unmarkOutput(*oldTensor);
		std::cout << "Done" << std::endl;
		// ==== Add Yolo Layer ====

		builder->setMaxBatchSize(1);
		config->setMaxWorkspaceSize(((size_t)1) << 30);
		config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

		if (USE_FP16)
			config->setFlag(nvinfer1::BuilderFlag::kFP16);

		samplesCommon::enableDLA(builder.get(), config.get(), DLA_CORE);

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
		if (!m_engine)
			throw(YoloTRTException("Failed to create Build Engine"));

		m_context = InferUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
		if (!m_context)
			throw(YoloTRTException("Failed to create Execution Context"));

		std::cout << "Writing engine file to disk ... " << std::flush;
		std::ofstream engineFile(ENGINE_FILE, std::ios::binary);
		if (!engineFile)
			throw(YoloTRTException(string_format("[SaveEngine] Failed to open Engine File: %s", ENGINE_FILE)));

		InferUniquePtr<nvinfer1::IHostMemory> pSerializedEngine{ m_engine->serialize() };
		if (!pSerializedEngine)
			throw(YoloTRTException("Failed to serialize Engine"));

		engineFile.write(static_cast<char *>(pSerializedEngine->data()), pSerializedEngine->size());
		engineFile.close();

		std::cout << "Done" << std::endl;
	}

	void processInput(const cv::Mat &image, const bool &scale = true) const
	{
		// ==== Pre-Process Image ====
		std::vector<float> data;
		cv::Mat scaledImg;
		cv::Mat imgConv;
		cv::Mat imgConv2;

		if (scale)
			cv::resize(image, scaledImg, cv::Size(m_yoloParser.GetWidth(), m_yoloParser.GetHeight()));
		else
			scaledImg = image;

		cv::cvtColor(scaledImg, imgConv, cv::COLOR_BGR2RGB);
		imgConv.convertTo(imgConv2, CV_32FC3, 1 / 255.0);

		std::vector<cv::Mat> channles(3);
		cv::split(imgConv2, channles);
		float *ptr1 = (float *)(channles[0].data);
		float *ptr2 = (float *)(channles[1].data);
		float *ptr3 = (float *)(channles[2].data);
		data.insert(data.end(), ptr1, ptr1 + m_yoloParser.GetWidth() * m_yoloParser.GetHeight());
		data.insert(data.end(), ptr2, ptr2 + m_yoloParser.GetWidth() * m_yoloParser.GetHeight());
		data.insert(data.end(), ptr3, ptr3 + m_yoloParser.GetWidth() * m_yoloParser.GetHeight());

		// ==== Pre-Process Image ====

		float *hostInputBuffer = static_cast<float *>(m_buffers->getHostBuffer(INPUT_LAYER));
		std::memcpy(hostInputBuffer, data.data(), data.size() * sizeof(float));
	}

	YoloResults nmsBoxes(YoloResults detections, const float &nmsThreshold) const
	{
		std::sort(std::begin(detections), std::end(detections), [](const YoloResult &a, const YoloResult &b) { return a.Conf() > b.Conf(); });
		YoloResults out;

		for (YoloResult &r : detections)
		{
			bool keep = true;
			for (const YoloResult &o : out)
			{
				if (keep)
				{
					YoloResult a = r;
					YoloResult b = o;
					a.Scale(m_imgSize.width, m_imgSize.height);
					b.Scale(m_imgSize.width, m_imgSize.height);
					float xx1 = std::max(a.x, b.x);
					float yy1 = std::max(a.y, b.y);
					float xx2 = std::min(a.x + a.w, b.x + b.w);
					float yy2 = std::min(a.y + a.h, b.y + b.h);

					float w   = std::max(0.0f, xx2 - xx1 + 1);
					float h   = std::max(0.0f, yy2 - yy1 + 1);
					float i   = w * h;
					float u   = (a.w * a.h) + (b.w * b.h) - i;
					float iou = i / u;

					keep = iou <= nmsThreshold;
				}
				else
					break;
			}
			if (keep)
			{
				// Make sure the box stays within the images boundaries
				if (r.x < 0.0f)
					r.x = 0.0f;
				if (r.y < 0.0f)
					r.y = 0.0f;
				if (r.w > m_imgSize.width - 1.0f)
					r.w = m_imgSize.width - 1.0f;
				if (r.h > m_imgSize.height - 1.0f)
					r.h = m_imgSize.height - 1.0f;
				out.push_back(r);
			}
		}

		return out;
	}

	YoloResults processOutput(const cv::Mat &img)
	{
		std::map<int32_t, YoloResults> results;
		YoloResults validResults;
		// cv::Mat imgLocal = img.clone();

		for (const auto &[outLayer, gridSize] : m_outputLayer)
		{
			const YoloResult *pResults = static_cast<const YoloResult *>(m_buffers->getHostBuffer(outLayer));

			// Print output values for each index
			for (int i = 0; i < gridSize * 3; i++)
			{
				const YoloResult &res = pResults[i];
				if (res.boxConfidence * res.classProb > m_threshold)
					results[static_cast<int32_t>(res.classID)].push_back(res);
			}
		}

		for (auto &[classID, results] : results)
		{
			YoloResults nms = nmsBoxes(results, 0.5f);
			validResults.insert(std::end(validResults), std::begin(nms), std::end(nms));
		}

		for (const YoloResult &r : validResults)
		{
			// cv::rectangle(imgLocal, cv::Point(r.x, r.y), cv::Point(r.x + r.w, r.y + r.h), cv::Scalar(0, 255, 0));
			// cv::putText(imgLocal, string_format("%s - %f", r.ClassName().c_str(), r.Conf()), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(255, 50, 50), 1);
		}

		// cv::imwrite("out.jpg", imgLocal);

		return validResults;
	}

	static bool fileExists(const std::string &name)
	{
		std::ifstream f(name);
		return f.good();
	}

private:
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
	InferUniquePtr<nvinfer1::IExecutionContext> m_context;
	std::shared_ptr<samplesCommon::BufferManager> m_buffers;

	std::map<std::string, uint32_t> m_outputLayer;

	bool m_verbose;
	cv::Size m_imgSize;
	YoloParser m_yoloParser;
	float m_threshold;
};
