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
from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from threading import RLock
from typing import ClassVar, Dict, List, Optional, cast, TYPE_CHECKING

from pyspark.accumulators import (
    Accumulator,
    AccumulatorParam,
    SpecialAccumulatorIds,
    _accumulatorRegistry,
    _deserialize_accumulator,
)
from pyspark.logger.logger import JSONFormatter, PySparkLoggerBase
from pyspark.sql.functions import from_json
from pyspark.sql.types import (
    ArrayType,
    MapType,
    StringType,
    StructType,
    TimestampNTZType,
)

if TYPE_CHECKING:
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.session import SparkSession

Logs = Dict[int, List[str]]

__all__ = ["PySparkUDFLogger"]


class _LogsParam(AccumulatorParam[Logs]):
    @staticmethod
    def zero(value: Logs) -> Logs:
        return defaultdict(list)

    @classmethod
    def addInPlace(cls, value1: Logs, value2: Logs) -> Logs:
        new_value = defaultdict(list)
        for k, v in value1.items():
            new_value[k].extend(v)
        for k, v in value2.items():
            new_value[k].extend(v)
        return new_value


LogsParam = _LogsParam()


class UDFLogHandler(logging.Handler):
    def __init__(self, result_id: int) -> None:
        super().__init__()
        self._accumulator = _deserialize_accumulator(
            SpecialAccumulatorIds.SQL_UDF_LOGGER, LogsParam.zero({}), LogsParam
        )
        formatter = JSONFormatter()
        formatter.default_msec_format = "%s.%03d"
        self._formatter = formatter

        self._result_id = result_id

    def emit(self, record: logging.LogRecord) -> None:
        if self._result_id is not None:
            msg = self._formatter.format(record)
            self._accumulator.add({self._result_id: [msg]})


class PySparkUDFLogger(PySparkLoggerBase):
    _result_id: ClassVar[Optional[int]] = None

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "PySparkUDFLogger", level=logging.WARN)
        if self._result_id is not None:
            self.addHandler(UDFLogHandler(self._result_id))

    @staticmethod
    def getLogger(name: Optional[str] = None) -> "PySparkUDFLogger":
        return PySparkUDFLogger(name)


class UDFLogCollector(ABC):
    LOG_ENTRY_SCHEMA: ClassVar[StructType] = (
        StructType()
        .add("ts", TimestampNTZType())
        .add("level", StringType())
        .add("logger", StringType())
        .add("msg", StringType())
        .add("context", MapType(StringType(), StringType()))
        .add(
            "exception",
            StructType()
            .add("class", StringType())
            .add("msg", StringType())
            .add("stacktrace", ArrayType(StringType())),
        )
    )

    def __init__(self) -> None:
        self._lock = RLock()

    @property
    @abstractmethod
    def _logs(self) -> Logs:
        pass

    def collect(self, id: int) -> Optional[List[str]]:
        with self._lock:
            return self._logs.get(id)

    def clear(self, id: Optional[int] = None) -> None:
        with self._lock:
            if id is not None:
                self._logs.pop(id, None)
            else:
                self._logs.clear()


class AccumulatorUDFLogCollector(UDFLogCollector):
    def __init__(self) -> None:
        super().__init__()
        if SpecialAccumulatorIds.SQL_UDF_LOGGER in _accumulatorRegistry:
            self._accumulator = _accumulatorRegistry[SpecialAccumulatorIds.SQL_UDF_LOGGER]
        else:
            self._accumulator = Accumulator(
                SpecialAccumulatorIds.SQL_UDF_LOGGER, LogsParam.zero({}), LogsParam
            )

    @property
    def _logs(self) -> Logs:
        return self._accumulator.value


class UDFLogs:
    def __init__(self, sparkSession: "SparkSession", collector: UDFLogCollector):
        self._sparkSession = sparkSession
        self._collector = collector

    def collect(self, id: int) -> Optional[List[str]]:
        """
        Collects the UDF logs for the given UDF ID.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        id : int, optional
            A UDF ID to be shown.

        Returns
        -------
        list of str
            The UDF logs as JSON format for the given ID.
        """
        return self._collector.collect(id)

    def collectAsDataFrame(self, id: int) -> Optional["DataFrame"]:
        """
        Collects the UDF logs for the given UDF ID.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        id : int, optional
            A UDF ID to be shown.

        Returns
        -------
        class:`DataFrame`
            The DataFrame containing UDF logs for the given ID.
        """
        logs = self._collector.collect(id)
        if logs is not None:
            return (
                self._sparkSession.createDataFrame([(row,) for row in logs], "json string")
                .select(from_json("json", self._collector.LOG_ENTRY_SCHEMA).alias("json"))
                .select("json.*")
            )
        else:
            return None

    def clear(self, id: Optional[int] = None) -> None:
        """
        Clear the UDF logs for the given UDF ID.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        id : int, optional
            The UDF ID whose UDF logs should be cleared.
            If not specified, all the UDF logs will be cleared.
        """
        self._collector.clear(id)
