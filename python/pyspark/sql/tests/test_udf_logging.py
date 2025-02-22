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
import logging
import unittest

from pyspark.sql import Row, functions as sf
from pyspark.sql.udf_logger import LogsT, PySparkUDFLogger
from pyspark.testing import assertDataFrameEqual
from pyspark.testing.sqlutils import ReusedSQLTestCase


class UDFLoggingTestsMixin:
    def setUp(self) -> None:
        super().setUp()
        self.spark.udf.logs.clear()

    @property
    def udf_logs(self) -> LogsT:
        return self.spark._udf_log_collector._logs

    @property
    def udf1(self):
        @sf.udf("string")
        def udf1(x) -> str:
            logger = PySparkUDFLogger.getLogger()
            logger.info(f"This is a log message: {x}")
            logger.warning(f"This is a log message: {x}", a=123)
            return str(logger.manager)

        return udf1.asNondeterministic()

    @property
    def udf2(self):
        @sf.udf("string")
        def udf2(x) -> str:
            logger = PySparkUDFLogger.getLogger()
            logger.setLevel(logging.INFO)  # Force to set the log level.
            logger.info(f"This is a log message: {x}", b="xyz")
            return str(logger.manager)

        return udf2.asNondeterministic()

    def test_udf_logging(self):
        udf1 = self.udf1
        udf2 = self.udf2

        df = self.spark.range(2, numPartitions=4).select(
            udf1(sf.col("id")).alias("udf1_1"),
            udf2(sf.col("id")).alias("udf2"),
            udf1(sf.col("id") * 1).alias("udf1_2"),
        )

        with self.sql_conf({"spark.sql.pyspark.udf.logging.maxEntries": -1}):
            df.collect()

        self.assertEqual(3, len(self.udf_logs), str(list(self.udf_logs)))

        udf1_1_id, udf2_id, udf1_2_id = sorted(self.udf_logs)

        for name, id, level, context in [
            (
                "udf1_1_id",
                udf1_1_id,
                "WARNING",
                {"udf_id": str(udf1_1_id), "a": "123", "udf_name": "udf1"},
            ),
            ("udf2_id", udf2_id, "INFO", {"udf_id": str(udf2_id), "b": "xyz", "udf_name": "udf2"}),
            (
                "udf1_2_id",
                udf1_2_id,
                "WARNING",
                {"udf_id": str(udf1_2_id), "a": "123", "udf_name": "udf1"},
            ),
        ]:
            with self.subTest(name=name):
                logs_df = self.spark.udf.logs.asDF(id)
                assertDataFrameEqual(
                    logs_df.select("level", "logger", "msg", "context"),
                    [
                        Row(
                            level=level,
                            logger="PySparkUDFLogger",
                            msg="This is a log message: 0",
                            context=context,
                        ),
                        Row(
                            level=level,
                            logger="PySparkUDFLogger",
                            msg="This is a log message: 1",
                            context=context,
                        ),
                    ],
                )
        assertDataFrameEqual(
            self.spark.udf.logs.asDF(),
            (
                self.spark.udf.logs.asDF(udf1_1_id)
                .union(self.spark.udf.logs.asDF(udf2_id))
                .union(self.spark.udf.logs.asDF(udf1_2_id))
            ),
        )

    def test_udf_log_max_entries(self):
        udf1 = self.udf1

        df = self.spark.range(10, numPartitions=4).select(udf1(sf.col("id")).alias("udf1_1"))

        # The default is 0
        df.collect()

        self.assertTrue(self.spark.udf.logs.asDF().isEmpty())

        self.spark.udf.logs.clear()

        # set to 5
        with self.sql_conf({"spark.sql.pyspark.udf.logging.maxEntries": 5}):
            df.collect()

        self.assertEqual(self.spark.udf.logs.asDF().count(), 5)

        self.spark.udf.logs.clear()

        # set to -1, which means no limit
        with self.sql_conf({"spark.sql.pyspark.udf.logging.maxEntries": -1}):
            df.collect()

        self.assertEqual(self.spark.udf.logs.asDF().count(), 10)

        # set to 2 without clear
        with self.sql_conf({"spark.sql.pyspark.udf.logging.maxEntries": 2}):
            df.collect()

        self.assertEqual(self.spark.udf.logs.asDF().count(), 2)

    def test_udf_log_level(self):
        udf1 = self.udf1
        udf2 = self.udf2

        df = self.spark.range(1, numPartitions=4).select(
            udf1(sf.col("id")).alias("udf1_1"), udf2(sf.col("id")).alias("udf2")
        )

        with self.sql_conf({"spark.sql.pyspark.udf.logging.maxEntries": -1}):
            df.collect()

            self.assertEqual(2, len(self.udf_logs), str(list(self.udf_logs)))
            udf1_id, udf2_id = sorted(self.udf_logs)

            assertDataFrameEqual(
                self.spark.udf.logs.asDF().select("level", "logger", "msg", "context"),
                [
                    Row(
                        level="WARNING",
                        logger="PySparkUDFLogger",
                        msg="This is a log message: 0",
                        context={"udf_id": str(udf1_id), "a": "123", "udf_name": "udf1"},
                    ),
                    Row(
                        level="INFO",
                        logger="PySparkUDFLogger",
                        msg="This is a log message: 0",
                        context={"udf_id": str(udf2_id), "b": "xyz", "udf_name": "udf2"},
                    ),
                ],
            )

            self.spark.udf.logs.clear()

            with self.sql_conf({"spark.sql.pyspark.udf.logging.logLevel": "INFO"}):
                df.collect()

            self.assertEqual(2, len(self.udf_logs), str(list(self.udf_logs)))
            udf1_id, udf2_id = sorted(self.udf_logs)

            assertDataFrameEqual(
                self.spark.udf.logs.asDF().select("level", "logger", "msg"),
                [
                    Row(
                        level="INFO",
                        logger="PySparkUDFLogger",
                        msg="This is a log message: 0",
                        context={"udf_id": str(udf1_id), "udf_name": "udf1"},
                    ),
                    Row(
                        level="WARNING",
                        logger="PySparkUDFLogger",
                        msg="This is a log message: 0",
                        context={"udf_id": str(udf1_id), "a": "123", "udf_name": "udf1"},
                    ),
                    Row(
                        level="INFO",
                        logger="PySparkUDFLogger",
                        msg="This is a log message: 0",
                        context={"udf_id": str(udf2_id), "b": "xyz", "udf_name": "udf2"},
                    ),
                ],
            )


class UDFLoggingTests(UDFLoggingTestsMixin, ReusedSQLTestCase):
    pass


if __name__ == "__main__":
    from pyspark.sql.tests.test_udf_logging import *  # noqa: F401

    try:
        import xmlrunner  # type: ignore

        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
