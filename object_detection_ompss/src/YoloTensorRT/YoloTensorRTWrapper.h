#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
	int InitVideoStream(const char* pStr);
	void InitYoloTensorRT(const char* pOnnxFile, const char* pConfigFile, const char* pEngineFile, const char* pClassFile, const int32_t dlaCore);
	void GetNextFrame(uint8_t* pData);
	void c2CvMat(uint8_t* pData, const int32_t height, const int32_t width, const uint8_t buffer);
	void ProcessNextFrame(const uint8_t buffer);
	void ProcessDetections(const uint8_t buffer);
	void Cleanup();
	void CheckFPS(uint32_t* pFrameCnt, const uint64_t iteration, const float maxFPS);

#ifdef __cplusplus
}
#endif
