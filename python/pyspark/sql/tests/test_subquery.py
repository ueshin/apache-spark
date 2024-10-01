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

import unittest

from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.testing import assertDataFrameEqual
from pyspark.testing.sqlutils import ReusedSQLTestCase


class SubqueryTestsMixin:
    @property
    def employees_df(self):
        # Define schema for employees DataFrame
        employees_schema = StructType(
            [
                StructField("employee_id", IntegerType(), nullable=False),
                StructField("name", StringType(), nullable=False),
                StructField("department_id", IntegerType(), nullable=False),
                StructField("salary", DoubleType(), nullable=False),
            ]
        )

        # Create data for employees DataFrame
        employees_data = [
            (101, "Alice", 1, 90000.0),
            (102, "Bob", 1, 80000.0),
            (103, "Charlie", 2, 70000.0),
            (104, "Diana", 2, 80000.0),
            (105, "Ethan", 3, 75000.0),
            (106, "Fiona", 3, 85000.0),
        ]

        # Create the employees DataFrame
        return self.spark.createDataFrame(employees_data, schema=employees_schema)

    @property
    def departments_df(self):
        # Define schema for departments DataFrame
        departments_schema = StructType(
            [
                StructField("department_id", IntegerType(), nullable=False),
                StructField("department_name", StringType(), nullable=False),
            ]
        )

        # Create data for departments DataFrame
        departments_data = [(1, "Engineering"), (2, "Sales"), (3, "Marketing")]

        # Create the departments DataFrame
        return self.spark.createDataFrame(departments_data, schema=departments_schema)

    @property
    def projects_df(self):
        # Define schema for projects DataFrame
        projects_schema = StructType(
            [
                StructField("project_id", IntegerType(), nullable=False),
                StructField("project_name", StringType(), nullable=False),
                StructField("department_id", IntegerType(), nullable=False),
            ]
        )

        # Create data for projects DataFrame
        projects_data = [(1, "Alpha", 1), (2, "Beta", 3), (3, "Gamma", 4)]

        # Create the projects DataFrame
        return self.spark.createDataFrame(projects_data, schema=projects_schema)

    def test_scalar_subquery(self):
        employees_df = self.employees_df
        departments_df = self.departments_df

        with self.tempView("employees", "departments"):
            employees_df.createOrReplaceTempView("employees")
            departments_df.createOrReplaceTempView("departments")

            actual1 = (
                employees_df.alias("e")
                .select(
                    "employee_id",
                    "name",
                    "salary",
                    "department_id",
                    employees_df.where(sf.col("department_id") == sf.col("e.department_id").outer())
                    .select(sf.avg("salary"))
                    .scalar()
                    .alias("department_avg_salary"),
                )
                .join(
                    departments_df.alias("d"),
                    sf.col("e.department_id") == sf.col("d.department_id"),
                )
                .select("employee_id", "name", "salary", "department_name", "department_avg_salary")
            )

            expected1 = self.spark.sql(
                """
                SELECT
                    e.employee_id,
                    e.name,
                    e.salary,
                    d.department_name,
                    (SELECT AVG(salary)
                     FROM employees
                     WHERE department_id = e.department_id) AS department_avg_salary
                FROM
                    employees e
                JOIN
                    departments d ON e.department_id = d.department_id
                """
            )

            assertDataFrameEqual(actual1, expected1)

            actual2 = (
                employees_df.alias("e")
                .where(
                    sf.col("salary")
                    > employees_df.where(
                        sf.col("department_id") == sf.col("e.department_id").outer()
                    )
                    .select(sf.avg("salary"))
                    .scalar()
                )
                .join(
                    departments_df.alias("d"),
                    sf.col("e.department_id") == sf.col("d.department_id"),
                )
                .select("employee_id", "name", "salary", "department_name")
            )

            expected2 = self.spark.sql(
                """
                SELECT
                    e.employee_id,
                    e.name,
                    e.salary,
                    d.department_name
                FROM
                    employees e
                JOIN
                    departments d ON e.department_id = d.department_id
                WHERE
                    e.salary > (SELECT AVG(salary)
                                FROM employees
                                WHERE department_id = e.department_id)
                """
            )

            assertDataFrameEqual(actual2, expected2)

    def test_exists(self):
        employees_df = self.employees_df
        projects_df = self.projects_df

        with self.tempView("employees", "projects"):
            employees_df.createOrReplaceTempView("employees")
            projects_df.createOrReplaceTempView("projects")

            actual1 = (
                employees_df.alias("e")
                .where(
                    projects_df.where(
                        sf.col("department_id") == sf.col("e.department_id").outer()
                    ).exists()
                )
                .select("employee_id", "name", "department_id")
            )

            expected1 = self.spark.sql(
                """
                SELECT
                    e.employee_id,
                    e.name,
                    e.department_id
                FROM
                    employees e
                WHERE
                    EXISTS (
                        SELECT 1
                        FROM projects p
                        WHERE p.department_id = e.department_id
                    )
                """
            )

            assertDataFrameEqual(actual1, expected1)

            actual2 = (
                employees_df.alias("e")
                .where(
                    ~projects_df.where(
                        sf.col("department_id") == sf.col("e.department_id").outer()
                    ).exists()
                )
                .select("employee_id", "name", "department_id")
            )

            expected2 = self.spark.sql(
                """
                SELECT
                    e.employee_id,
                    e.name,
                    e.department_id
                FROM
                    employees e
                WHERE
                    NOT EXISTS (
                        SELECT 1
                        FROM projects p
                        WHERE p.department_id = e.department_id
                    )
                """
            )

            assertDataFrameEqual(actual2, expected2)


class SubqueryTests(SubqueryTestsMixin, ReusedSQLTestCase):
    pass


if __name__ == "__main__":
    from pyspark.sql.tests.test_subquery import *  # noqa: F401

    try:
        import xmlrunner  # type: ignore

        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
