#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections import defaultdict
import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

from pyspark.accumulators import (
    AccumulatorParam,
    SpecialAccumulatorIds,
    _deserialize_accumulator,
)
from pyspark.logger.logger import JSONFormatter, PySparkLoggerBase

Logs = Dict[int, List[str]]
LogsParamValue = Tuple[Logs, Optional[int]]

__all__ = ["PySparkUDFLogger"]


class LogsParam(AccumulatorParam[LogsParamValue]):
    def __init__(self, max_entries: Optional[int]):
        super().__init__()
        self._max_entries = max_entries

    def zero(self, value: LogsParamValue) -> LogsParamValue:
        return defaultdict(list), self._max_entries

    def addInPlace(self, logs1: LogsParamValue, logs2: LogsParamValue) -> LogsParamValue:
        value1, _ = logs1
        value2, me2 = logs2
        me = self._max_entries
        if me is None:
            me = me2
            if me is None:
                me = -1
        for k, v in value2.items():
            if me == 0:
                value1[k] = []
            else:
                value1[k].extend(v)
                if me > 0:
                    value1[k] = value1[k][-me:]
        return value1, self._max_entries


class UDFLogHandler(logging.Handler):
    def __init__(self, result_id: int, max_entries: int, default_context: Dict[str, Any]) -> None:
        super().__init__()
        self._result_id = result_id

        param = LogsParam(max_entries)
        self._accumulator = _deserialize_accumulator(
            SpecialAccumulatorIds.SQL_UDF_LOGGER, param.zero(({}, None)), param
        )
        formatter = JSONFormatter()
        formatter.default_msec_format = "%s.%03d"
        self._formatter = formatter

        if "udf_id" not in default_context:
            default_context["udf_id"] = result_id
        self._default_context = default_context

    def emit(self, record: logging.LogRecord) -> None:
        if self._result_id is not None:
            context = record.__dict__.get("kwargs", {})
            for k, v in self._default_context.items():
                if context.get(k) is None:
                    context[k] = v
            record.__dict__["kwargs"] = context
            msg = self._formatter.format(record)
            self._accumulator.add(({self._result_id: [msg]}, None))


class PySparkUDFLogger(PySparkLoggerBase):
    _max_entries: ClassVar[int] = 0
    _log_level: ClassVar[Union[int, str]] = logging.WARN
    _result_id: ClassVar[Optional[int]] = None

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "PySparkUDFLogger", level=self._log_level)
        if self._result_id is not None:
            self.addHandler(
                UDFLogHandler(self._result_id, self._max_entries, {"udf_id": self._result_id})
            )

    @staticmethod
    def getLogger(name: Optional[str] = None) -> "PySparkUDFLogger":
        return PySparkUDFLogger(name)
