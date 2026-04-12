from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pathlib import Path

spark = SparkSession.builder \
    .appName("clickstream-feature-engineering") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

sessions_path = Path(__file__).parent.parent / "output" / "sessions.parquet"
output_path = Path(__file__).parent.parent / "output" / "features.parquet"

df = spark.read.parquet(str(sessions_path))

df = df.withColumn("month", F.month("date")) \
       .withColumn("day_of_week", F.dayofweek("date"))

df = df.withColumn("label", F.col("bought").cast("int")).drop("bought")

feature_cols = ["n_clicks", "avg_price", "n_categories", "n_colours", "country", "month", "day_of_week"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
df = scaler.fit(df).transform(df).drop("features_raw")

df.select("session_id", "features", "label").write.parquet(str(output_path), mode="overwrite")

print("Done. Schema:")
df.printSchema()
print(f"Total rows: {df.count()}")