from pyspark.sql import SparkSession
from pathlib import Path

spark = SparkSession.builder \
    .appName("clickstream-ingest") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

data_path = Path(__file__).parent.parent / "data" / "e-shop_clothing_2008.csv"

df = spark.read.csv(
    str(data_path),
    header=True,
    inferSchema=True,
    sep=";"
)

df.printSchema()
df.show(5)
print(f"Total rows: {df.count()}")