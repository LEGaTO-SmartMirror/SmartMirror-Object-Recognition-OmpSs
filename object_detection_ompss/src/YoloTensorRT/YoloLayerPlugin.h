#ifndef YOLO_LAYER_PLUGIN_H
#define YOLO_LAYER_PLUGIN_H

#include <NvInfer.h>
#include <string>
#include <vector>

#define MAX_ANCHORS 6

namespace Yolo
{
struct alignas(float) Detection
{
	float bbox[4]; // x, y, w, h
	float det_confidence;
	float class_id;
	float class_confidence;
};
} // namespace Yolo

namespace nvinfer1
{
class YoloLayerPlugin : public IPluginV2IOExt
{
public:
	YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y);
	YoloLayerPlugin(const void* data, size_t length);

	~YoloLayerPlugin() override = default;

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

	int initialize() override;

	void terminate() override;

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() const override;

	virtual void serialize(void* buffer) const override;

	bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
	{
		return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
	}

	const char* getPluginType() const override;

	const char* getPluginVersion() const override;

	void destroy() override;

	IPluginV2IOExt* clone() const override;

	void setPluginNamespace(const char* pluginNamespace) override;

	const char* getPluginNamespace() const override;

	DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

	bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

	bool canBroadcastInputAcrossBatch(int inputIndex) const override;

	void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

	void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override TRTNOEXCEPT;

	void detachFromContext() override;

private:
	using IPluginV2IOExt::configurePlugin;

private:
	void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

	int mThreadCount = 64;
	int mYoloWidth, mYoloHeight, mNumAnchors;
	float mAnchorsHost[MAX_ANCHORS * 2];
	float* mAnchors; // allocated on GPU
	int mNumClasses;
	int mInputWidth, mInputHeight;
	float mScaleXY;

	const char* mPluginNamespace;
};

class YoloPluginCreator : public IPluginCreator
{
public:
	YoloPluginCreator();

	~YoloPluginCreator() override = default;

	const char* getPluginName() const override;

	const char* getPluginVersion() const override;

	const PluginFieldCollection* getFieldNames() override;

	IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

	IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

	void setPluginNamespace(const char* libNamespace) override
	{
		mNamespace = libNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mNamespace.c_str();
	}

private:
	static PluginFieldCollection mFC;
	static std::vector<nvinfer1::PluginField> mPluginAttributes;
	std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

}; // namespace nvinfer1

#endif // YOLO_LAYER_PLUGIN_H
