from pyspark.sql import SparkSession, functions as F
from pathlib import Path

spark = SparkSession.builder \
    .appName("clickstream-sessionize")\
    .master("local[*]")\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

cleaned_path = Path(__file__).parent.parent / "output" / "cleaned.parquet"
output_path = Path(__file__).parent.parent / "output" / "sessions.parquet"

df = spark.read.parquet(str(cleaned_path))

sessions = df.groupBy("session_id").agg(
    F.count("*").alias("n_clicks"),
    F.avg("price").alias("avg_price"),
    F.countDistinct("page_1_main_category").alias("n_categories"),
    F.countDistinct("colour").alias("n_colours"),
    F.max("price_2").alias("bought"),
    F.first("country").alias("country"),
    F.first("date").alias("date")
    )

sessions.write.parquet(str(output_path), mode="overwrite")

print("Done. Schema:")
sessions.printSchema()
print(f"Total sessions: {sessions.count()}")
