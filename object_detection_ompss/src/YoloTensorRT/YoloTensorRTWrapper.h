#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

	int InitVideoStream(const char *pStr);
	void InitYoloTensorRT();
	uint8_t *GetNextFrame();
	void ProcessNextFrame(uint8_t *pData, const int32_t height, const int32_t width);
	void ProcessDetections();
	void Cleanup();
	void CheckFPS();

#ifdef __cplusplus
}
#endif
