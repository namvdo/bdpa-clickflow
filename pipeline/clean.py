from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType
from pathlib import Path
import re

spark = SparkSession.builder \
    .appName("clickstream-clean") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

data_path = Path(__file__).parent.parent / "data" / "e-shop_clothing_2008.csv"
output_path = Path(__file__).parent.parent / "output" / "cleaned.parquet"

df = spark.read.csv(
    str(data_path),
    header=True,
    inferSchema=True,
    sep=";"
)

for col in df.columns:
    new_name = re.sub(r'[ ()/]+', '_', col).strip('_').lower()
    df = df.withColumnRenamed(col, new_name)

df = df.withColumn(
    "date",
    F.to_date(F.concat_ws("-", F.col("year"), F.col("month"), F.col("day")), "yyyy-M-d")
).drop("year", "month", "day")

for col in ["country", "page_1_main_category", "colour", "location", "model_photography"]:
    df = df.withColumn(col, F.col(col).cast(StringType()))

df = df.withColumn("price_2", F.col("price_2") == 1)

df.write.parquet(str(output_path), mode="overwrite")

print("Done. Schema:")
df.printSchema()
print(f"Total rows: {df.count()}")