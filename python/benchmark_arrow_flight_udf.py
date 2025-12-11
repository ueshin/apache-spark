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

"""
Comprehensive benchmark comparing standard Arrow UDFs vs Arrow Flight UDFs.

This benchmark measures:
1. Data transfer performance
2. Memory usage
3. Throughput for different data sizes
4. Latency characteristics
5. Scalability with multiple UDFs
"""

import time
import gc
import psutil
from typing import Iterator, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import arrow_udf, col, lit, rand, when
from pyspark.sql.types import IntegerType, StringType, DoubleType, LongType


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    test_name: str
    implementation: str  # "standard" or "flight"
    data_size: int
    execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    throughput_rows_per_sec: float
    cpu_percent: float
    additional_metrics: Dict[str, Any]


class PerformanceMonitor:
    """Monitor system performance during benchmark execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        gc.collect()  # Clean up before measurement

    def sample_metrics(self):
        """Sample current performance metrics."""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()

            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)
        except Exception:
            pass  # Ignore sampling errors

    def get_results(self, execution_time: float) -> Dict[str, float]:
        """Get performance monitoring results."""
        if not self.memory_samples:
            return {"memory_peak_mb": 0.0, "memory_avg_mb": 0.0, "cpu_percent": 0.0}

        return {
            "memory_peak_mb": max(self.memory_samples),
            "memory_avg_mb": sum(self.memory_samples) / len(self.memory_samples),
            "cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples)
            if self.cpu_samples
            else 0.0,
        }


class ArrowUDFBenchmark:
    """Comprehensive benchmark suite for Arrow UDFs."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.monitor = PerformanceMonitor()

    def create_spark_session(self, use_flight: bool = False) -> SparkSession:
        """Create Spark session with appropriate configuration."""
        builder = (
            SparkSession.builder.appName(
                f"ArrowUDFBenchmark_{'Flight' if use_flight else 'Standard'}"
            )
            .config("spark.sql.execution.pythonUDF.arrow.enabled", "true")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
            .config("spark.sql.execution.pyspark.udf.idleTimeoutSeconds", 5)
            .config("spark.sql.execution.pyspark.udf.tracebackDumpIntervalSeconds", 5)
        )
        return builder.getOrCreate()

    @contextmanager
    def timed_execution(self, test_name: str, implementation: str, data_size: int):
        """Context manager for timed execution with performance monitoring."""
        self.monitor.start_monitoring()
        start_time = time.time()

        try:
            yield
        finally:
            execution_time = time.time() - start_time
            perf_metrics = self.monitor.get_results(execution_time)

            throughput = data_size / execution_time if execution_time > 0 else 0

            result = BenchmarkResult(
                test_name=test_name,
                implementation=implementation,
                data_size=data_size,
                execution_time=execution_time,
                memory_peak_mb=perf_metrics["memory_peak_mb"],
                memory_avg_mb=perf_metrics["memory_avg_mb"],
                throughput_rows_per_sec=throughput,
                cpu_percent=perf_metrics["cpu_percent"],
                additional_metrics={},
            )

            self.results.append(result)

    def create_test_data(self, spark: SparkSession, size: int, complexity: str = "simple"):
        """Create test data of specified size and complexity."""
        if complexity == "simple":
            # Simple string and numeric data
            return spark.range(size).select(
                col("id"),
                (col("id") % 1000).cast("string").alias("text"),
                (col("id") * 1.5).cast("double").alias("value"),
                (col("id") % 100).cast("int").alias("category"),
            )
        elif complexity == "medium":
            # More complex data with longer strings
            return spark.range(size).select(
                col("id"),
                concat_ws("_", lit("prefix"), col("id"), lit("suffix")).alias("text"),
                (rand() * 1000).alias("value"),
                (col("id") % 10).cast("int").alias("category"),
                when(col("id") % 2 == 0, "even").otherwise("odd").alias("parity"),
            )
        else:  # complex
            # Complex data with nested structures
            return spark.range(size).select(
                col("id"),
                concat_ws(
                    "-", lit("complex"), col("id"), lit("data"), (col("id") % 1000).cast("string")
                ).alias("text"),
                (rand() * col("id")).alias("value1"),
                (rand() * 100).alias("value2"),
                (col("id") % 50).cast("int").alias("category"),
                array(col("id"), col("id") * 2, col("id") * 3).alias("numbers"),
            )

    def benchmark_string_operations(self, spark: SparkSession, data_size: int, implementation: str):
        """Benchmark string processing operations."""

        @arrow_udf(returnType=IntegerType())
        def string_length_udf(strings: pa.Array) -> pa.Array:
            return pc.utf8_length(strings)

        @arrow_udf(returnType=StringType())
        def string_upper_udf(strings: pa.Array) -> pa.Array:
            return pc.ascii_upper(strings)

        @arrow_udf(returnType=StringType())
        def string_reverse_udf(strings: pa.Array) -> pa.Array:
            # Custom reverse operation
            result = []
            for i in range(len(strings)):
                if strings[i].is_valid:
                    result.append(strings[i].as_py()[::-1])
                else:
                    result.append(None)
            return pa.array(result)

        df = self.create_test_data(spark, data_size, "simple")

        with self.timed_execution("string_length", implementation, data_size):
            result = df.select(string_length_udf(col("text")).alias("length")).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("string_upper", implementation, data_size):
            result = df.select(string_upper_udf(col("text")).alias("upper")).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("string_reverse", implementation, data_size):
            result = df.select(string_reverse_udf(col("text")).alias("reversed")).collect()
            self.monitor.sample_metrics()

    def benchmark_numeric_operations(
        self, spark: SparkSession, data_size: int, implementation: str
    ):
        """Benchmark numeric processing operations."""

        @arrow_udf(returnType=DoubleType())
        def math_complex_udf(values: pa.Array) -> pa.Array:
            # Complex math: sqrt(x^2 + 1) * log(x + 1)
            x_squared = pc.multiply(values, values)
            plus_one = pc.add(x_squared, pa.scalar(1.0))
            sqrt_part = pc.sqrt(plus_one)
            log_part = pc.ln(pc.add(values, pa.scalar(1.0)))
            return pc.multiply(sqrt_part, log_part)

        @arrow_udf(returnType=DoubleType())
        def statistical_udf(values: pa.Array) -> pa.Array:
            # Statistical operations
            mean_val = pc.mean(values)
            variance = pc.variance(values)
            return pc.add(pc.subtract(values, mean_val), variance)

        @arrow_udf(returnType=DoubleType())
        def aggregation_udf(values: pa.Array) -> pa.Array:
            # Aggregation-like operations
            cumsum = pc.cumulative_sum(pc.cast(values, pa.float64()))
            return cumsum

        df = self.create_test_data(spark, data_size, "simple")

        with self.timed_execution("math_complex", implementation, data_size):
            result = df.select(math_complex_udf(col("value")).alias("complex_math")).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("statistical", implementation, data_size):
            result = df.select(statistical_udf(col("value")).alias("stats")).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("aggregation", implementation, data_size):
            result = df.select(aggregation_udf(col("value")).alias("agg")).collect()
            self.monitor.sample_metrics()

    def benchmark_multi_column_operations(
        self, spark: SparkSession, data_size: int, implementation: str
    ):
        """Benchmark operations involving multiple columns."""

        @arrow_udf(returnType=DoubleType())
        def multi_column_math_udf(col1: pa.Array, col2: pa.Array, col3: pa.Array) -> pa.Array:
            # Complex multi-column operation
            product = pc.multiply(col1, col2)
            ratio = pc.divide(product, pc.add(col3, pa.scalar(1.0)))
            return pc.sqrt(pc.abs(ratio))

        @arrow_udf(returnType=StringType())
        def multi_column_string_udf(text: pa.Array, nums: pa.Array) -> pa.Array:
            # Combine text and numbers
            result = []
            for i in range(len(text)):
                if text[i].is_valid and nums[i].is_valid:
                    result.append(f"{text[i].as_py()}_{nums[i].as_py()}")
                else:
                    result.append(None)
            return pa.array(result)

        df = self.create_test_data(spark, data_size, "simple")

        with self.timed_execution("multi_column_math", implementation, data_size):
            result = df.select(
                multi_column_math_udf(
                    col("value"), col("id").cast("double"), col("category").cast("double")
                ).alias("multi_math")
            ).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("multi_column_string", implementation, data_size):
            result = df.select(
                multi_column_string_udf(col("text"), col("category")).alias("combined")
            ).collect()
            self.monitor.sample_metrics()

    def benchmark_iterator_operations(
        self, spark: SparkSession, data_size: int, implementation: str
    ):
        """Benchmark iterator-based UDF operations."""

        @arrow_udf(returnType=DoubleType())
        def batch_processing_udf(iterator: Iterator[pa.Array]) -> Iterator[pa.Array]:
            batch_stats = []
            for batch in iterator:
                # Compute batch statistics
                batch_mean = pc.mean(batch)
                normalized = pc.subtract(batch, batch_mean)
                batch_stats.append(batch_mean.as_py())
                yield normalized

        @arrow_udf(returnType=LongType())
        def stateful_processing_udf(iterator: Iterator[pa.Array]) -> Iterator[pa.Array]:
            running_sum = 0
            for batch in iterator:
                # Stateful processing across batches
                batch_sum = pc.sum(batch).as_py()
                running_sum += batch_sum

                # Add running sum to each element
                result = pc.add(pc.cast(batch, pa.int64()), pa.scalar(running_sum))
                yield result

        df = self.create_test_data(spark, data_size, "simple")

        with self.timed_execution("batch_processing", implementation, data_size):
            result = df.select(batch_processing_udf(col("value")).alias("batch_proc")).collect()
            self.monitor.sample_metrics()

        with self.timed_execution("stateful_processing", implementation, data_size):
            result = df.select(stateful_processing_udf(col("category")).alias("stateful")).collect()
            self.monitor.sample_metrics()

    def benchmark_data_size_scaling(self, spark: SparkSession, implementation: str):
        """Benchmark performance scaling with different data sizes."""
        data_sizes = [1000, 10000, 100000, 500000, 1000000]

        @arrow_udf(returnType=DoubleType())
        def scaling_test_udf(values: pa.Array) -> pa.Array:
            return pc.multiply(pc.sqrt(values), pa.scalar(2.0))

        for size in data_sizes:
            df = self.create_test_data(spark, size, "simple")

            with self.timed_execution(f"scaling_test_{size}", implementation, size):
                result = df.select(scaling_test_udf(col("value")).alias("scaled")).collect()
                self.monitor.sample_metrics()

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark comparing standard vs Flight implementations."""
        print("Starting Comprehensive Arrow UDF Benchmark")
        print("=" * 60)

        # Test configurations
        data_sizes = [10000, 100000, 500000]
        implementations = ["standard", "flight"]

        spark = self.create_spark_session()

        for implementation in implementations:
            print(f"\nTesting {implementation.upper()} implementation...")

            use_flight = implementation == "flight"

            if use_flight:
                spark.conf.set("spark.sql.execution.pythonUDF.arrow.flight.enabled", "true")
            else:
                spark.conf.set("spark.sql.execution.pythonUDF.arrow.flight.enabled", "false")

            try:
                for data_size in data_sizes:
                    print(f"  Data size: {data_size:,} rows")

                    # Run different benchmark categories
                    self.benchmark_string_operations(spark, data_size, implementation)
                    self.benchmark_numeric_operations(spark, data_size, implementation)
                    self.benchmark_multi_column_operations(spark, data_size, implementation)

                    # if data_size <= 100000:  # Iterator tests only for smaller datasets
                    #     self.benchmark_iterator_operations(spark, data_size, implementation)

                # Run scaling test
                self.benchmark_data_size_scaling(spark, implementation)

            finally:
                time.sleep(2)  # Allow cleanup

        spark.stop()

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("Arrow UDF Performance Benchmark Report")
        report.append("=" * 50)
        report.append("")

        # Group results by test name and data size
        grouped_results = {}
        for result in self.results:
            key = (result.test_name, result.data_size)
            if key not in grouped_results:
                grouped_results[key] = {}
            grouped_results[key][result.implementation] = result

        # Performance comparison table
        report.append("Performance Comparison Summary")
        report.append("-" * 40)
        report.append(
            f"{'Test Name':<20} {'Data Size':<12} {'Standard (s)':<12} {'Flight (s)':<12} {'Speedup':<10} {'Memory Δ':<12}"
        )
        report.append("-" * 90)

        total_standard_time = 0
        total_flight_time = 0
        speedup_improvements = []
        memory_improvements = []

        for (test_name, data_size), implementations in sorted(grouped_results.items()):
            if "standard" in implementations and "flight" in implementations:
                std_result = implementations["standard"]
                flight_result = implementations["flight"]

                speedup = (
                    std_result.execution_time / flight_result.execution_time
                    if flight_result.execution_time > 0
                    else 0
                )
                memory_delta = (
                    (
                        (flight_result.memory_peak_mb - std_result.memory_peak_mb)
                        / std_result.memory_peak_mb
                        * 100
                    )
                    if std_result.memory_peak_mb > 0
                    else 0
                )

                speedup_improvements.append(speedup)
                memory_improvements.append(memory_delta)
                total_standard_time += std_result.execution_time
                total_flight_time += flight_result.execution_time

                report.append(
                    f"{test_name:<20} {data_size:<12,} {std_result.execution_time:<12.3f} {flight_result.execution_time:<12.3f} {speedup:<10.2f}x {memory_delta:<12.1f}%"
                )

        report.append("-" * 90)

        # Summary statistics
        if speedup_improvements:
            avg_speedup = sum(speedup_improvements) / len(speedup_improvements)
            max_speedup = max(speedup_improvements)
            min_speedup = min(speedup_improvements)

            avg_memory_change = sum(memory_improvements) / len(memory_improvements)

            overall_speedup = (
                total_standard_time / total_flight_time if total_flight_time > 0 else 0
            )

            report.append("")
            report.append("Summary Statistics")
            report.append("-" * 20)
            report.append(f"Overall Speedup: {overall_speedup:.2f}x")
            report.append(f"Average Speedup: {avg_speedup:.2f}x")
            report.append(f"Best Speedup: {max_speedup:.2f}x")
            report.append(f"Worst Speedup: {min_speedup:.2f}x")
            report.append(f"Average Memory Change: {avg_memory_change:.1f}%")
            report.append("")

        # Detailed results by category
        categories = {}
        for result in self.results:
            category = result.test_name.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, results in categories.items():
            report.append(f"\n{category.title()} Operations Detailed Results")
            report.append("-" * 40)

            for result in sorted(results, key=lambda x: (x.data_size, x.implementation)):
                throughput_k = result.throughput_rows_per_sec / 1000
                report.append(
                    f"  {result.implementation:<8} | {result.data_size:<8,} rows | "
                    f"{result.execution_time:<8.3f}s | {throughput_k:<8.1f}k rows/s | "
                    f"{result.memory_peak_mb:<8.1f}MB | {result.cpu_percent:<6.1f}% CPU"
                )

        # Scaling analysis
        report.append("\nScaling Analysis")
        report.append("-" * 20)

        scaling_results = [r for r in self.results if r.test_name.startswith("scaling_test_")]
        if scaling_results:
            for impl in ["standard", "flight"]:
                impl_results = [r for r in scaling_results if r.implementation == impl]
                if len(impl_results) >= 2:
                    impl_results.sort(key=lambda x: x.data_size)

                    # Calculate scaling factor (how execution time grows with data size)
                    small = impl_results[0]
                    large = impl_results[-1]

                    size_ratio = large.data_size / small.data_size
                    time_ratio = large.execution_time / small.execution_time
                    scaling_efficiency = size_ratio / time_ratio

                    report.append(f"{impl.title()} scaling efficiency: {scaling_efficiency:.2f}")
                    report.append(
                        f"  {small.data_size:,} -> {large.data_size:,} rows: "
                        f"{small.execution_time:.3f}s -> {large.execution_time:.3f}s"
                    )

        return "\n".join(report)

    def save_detailed_results(self, filename: str = "arrow_udf_benchmark_results.csv"):
        """Save detailed results to CSV file."""
        import csv

        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "test_name",
                "implementation",
                "data_size",
                "execution_time",
                "memory_peak_mb",
                "memory_avg_mb",
                "throughput_rows_per_sec",
                "cpu_percent",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(
                    {
                        "test_name": result.test_name,
                        "implementation": result.implementation,
                        "data_size": result.data_size,
                        "execution_time": result.execution_time,
                        "memory_peak_mb": result.memory_peak_mb,
                        "memory_avg_mb": result.memory_avg_mb,
                        "throughput_rows_per_sec": result.throughput_rows_per_sec,
                        "cpu_percent": result.cpu_percent,
                    }
                )

        print(f"Detailed results saved to {filename}")


def main():
    """Run the comprehensive benchmark."""
    print("Arrow UDF vs Arrow Flight UDF Performance Benchmark")
    print("=" * 60)
    print("This benchmark compares the performance of standard Arrow UDFs")
    print("with the new Arrow Flight-based implementation.")
    print("")

    benchmark = ArrowUDFBenchmark()

    try:
        benchmark.run_comprehensive_benchmark()

        # Generate and display report
        report = benchmark.generate_report()
        print("\n" + report)

        # Save detailed results
        benchmark.save_detailed_results()

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
