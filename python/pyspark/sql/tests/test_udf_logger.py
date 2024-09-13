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
from pyspark.sql.udf_logger import Logs, PySparkUDFLogger
from pyspark.testing import assertDataFrameEqual
from pyspark.testing.sqlutils import ReusedSQLTestCase


class UDFLoggerTestsMixin:
    def setUp(self) -> None:
        super().setUp()
        self.spark.udfLogs.clear()

    @property
    def udf_logs(self) -> Logs:
        return self.spark._udf_log_collector._logs

    def test_udf_logging(self):
        @sf.udf("string")
        def udf1(x) -> str:
            logger = PySparkUDFLogger.getLogger()
            logger.info(f"This is a log message: {x}")
            logger.warning(f"This is a log message: {x}", a=123)
            return str(logger.manager)

        @sf.udf("string")
        def udf2(x) -> str:
            logger = PySparkUDFLogger.getLogger()
            logger.setLevel(logging.INFO)
            logger.info(f"This is a log message: {x}", b="xyz")
            return str(logger.manager)

        df = self.spark.range(2).select(
            udf1(sf.col("id")).alias("udf1_1"),
            udf2(sf.col("id")).alias("udf2"),
            udf1(sf.col("id") * 1).alias("udf1_2"),
        )
        df.collect()

        self.assertEqual(3, len(self.udf_logs), str(list(self.udf_logs)))

        udf1_1_id, udf2_id, udf1_2_id = sorted(self.udf_logs)

        for name, id, level, context in [
            ("udf1_1_id", udf1_1_id, "WARNING", {"a": "123"}),
            ("udf2_id", udf2_id, "INFO", {"b": "xyz"}),
            ("udf1_2_id", udf1_2_id, "WARNING", {"a": "123"}),
        ]:
            with self.subTest(name=name):
                logs = self.spark.udfLogs.collect(id)
                self.assertEqual(2, len(logs))

                logs_df = self.spark.udfLogs.collectAsDataFrame(id)
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


class UDFLoggerTests(UDFLoggerTestsMixin, ReusedSQLTestCase):
    pass


if __name__ == "__main__":
    from pyspark.sql.tests.test_udf_logger import *  # noqa: F401

    try:
        import xmlrunner  # type: ignore

        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
