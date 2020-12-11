#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
	typedef struct yoloTRT YoloTRT_t;

	int InitVideoStream(const char *pStr);
	void InitYoloTensorRT();
	void GetNextFrame(uint8_t *pData);
	void c2CvMat(uint8_t *pData, const int32_t height, const int32_t width, const uint8_t buffer);
	void ProcessNextFrame(const uint8_t buffer);
	void ProcessDetections(const uint8_t buffer);
	void Cleanup();
	void CheckFPS(uint32_t *pFrameCnt);

#ifdef __cplusplus
}
#endif
