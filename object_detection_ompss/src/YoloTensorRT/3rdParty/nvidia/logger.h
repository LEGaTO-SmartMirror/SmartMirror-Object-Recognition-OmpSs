/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "logging.h"

namespace TrtLog
{
static Logger gLogger{ Logger::Severity::kINFO };
static LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
static LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
static LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
static LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
static LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

void setReportableSeverity(Logger::Severity severity)
{
	gLogger.setReportableSeverity(severity);
	gLogVerbose.setReportableSeverity(severity);
	gLogInfo.setReportableSeverity(severity);
	gLogWarning.setReportableSeverity(severity);
	gLogError.setReportableSeverity(severity);
	gLogFatal.setReportableSeverity(severity);
}
} // namespace TrtLog

#endif // LOGGER_H
