from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler, MinMaxScaler, StringIndexer, PCA
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("ML").master("local[*]").getOrCreate()
spark.conf.set("spark.sql.ansi.enabled", False)
data_dir = "../dataset/e-shop clothing 2008.csv"

eu_codes = [
    2,
    3,
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    18,
    21,
    22,
    23,
    24,
    25,
    27,
    30,
    34,
    35,
    36,
    37,
    41,
]
non_eu_europe_codes = [7, 19, 28, 31, 32, 33, 38, 39]
outside_europe_codes = [1, 4, 5, 6, 20, 26, 40, 42]
unidentified_codes = [12, 43, 44, 45, 46, 47]


def pipeline(drop: list[str] = ["year", "month", "order", "session ID"] ):
    '''
    Returns Dataframe and pipeline objects\n
    Splits the "page 2 (clothing model)" into model_letter and model_number\n
    PCA is the last stage\n
    '''

    df = spark.read.csv(data_dir, inferSchema=True, header=True, sep=";")
    df = df.drop(*drop)
    # add region column based on country codes
    df = df.withColumn("country", F.when(F.col("country") == 29, 0)
        .when(F.col("country").isin(eu_codes), 1)
        .when(F.col("country").isin(non_eu_europe_codes), 2)
        .when(F.col("country").isin(outside_europe_codes), 3)
        .otherwise(4))

    #split clothing model strings into letter and number
    string_col = "page 2 (clothing model)"
    _df = df.withColumn("model_letter", F.substring(string_col, 1, 1)).withColumn("model_number", F.substring(string_col, 2, 3)).drop(string_col)
    _df = _df.withColumnRenamed(_df.columns[1], "main_category")

    #prepare column names
    numeric_cols = ["day", "price"]
    string_cols = ["model_letter", "model_number"]
    binary_cols = ["price 2", "model photography"]

    si_output = [col + "_id" for col in string_cols]
    binary_output = [col + "_id" for col in binary_cols]

    categorical_cols = [col for col in _df.columns if col not in numeric_cols + string_cols + binary_cols] + si_output
    final_cols = [col+ "_enc"  for col in categorical_cols] + [col + "_s" for col in numeric_cols] + binary_output

    #first encode letters, and additionally convert binary to 0, 1 index (1,2 now)
    si = StringIndexer(inputCols=string_cols + binary_cols, outputCols=si_output + binary_output)

    #onehot categorical columns
    ohe = OneHotEncoder(inputCols=categorical_cols, outputCols=[col + "_enc" for col in categorical_cols])

    #va's for continous, vectorize continous
    va_1 = VectorAssembler(inputCols=["price"], outputCol="price_v")
    va_2 = VectorAssembler(inputCols=["day"], outputCol="day_v")
    vas = [va_1, va_2]

    # scalers 
    ss = StandardScaler(inputCol = "price_v", outputCol="price_s")
    mm = MinMaxScaler(inputCol= "day_v", outputCol="day_s")

    #Final vector 
    va_final = VectorAssembler(inputCols=final_cols, outputCol="vector")

    #Reduce
    pca = PCA(k = 30, inputCol="vector", outputCol="reduced")

    stages = [*vas, si, ohe, ss, mm, va_final, pca]
    pipe = Pipeline(stages = stages)
    _df.show(5)

    return _df, pipe

def evaluate_clustering_model(model_object, df: DataFrame, N_K: list[int], features_col: str = "reduced", prediction_col: str = "pred", plot_results: bool = False):
    '''
    For models with n-clusters\n
    Since dataset is small we can straight evaluate also DB and CH scores using sklearn, by converting our feature vector into numpy array,\n
    Bigger dataset requires subsampling or implemenation of scoring\n
    '''
    X = df
    x = X.select(vector_to_array(X[features_col]).alias("vector")).toPandas()
    x = np.array(x["vector"].to_list())

    results = {"N_K": [], "silouhette" : [], "davies_bouldin": [], "calinski_harabasz": []}
    ce = ClusteringEvaluator(predictionCol=prediction_col, featuresCol=features_col)
    model_object.setFeaturesCol(features_col)
    model_object.setPredictionCol(prediction_col)
    model_object.setSeed(55)

    for K in N_K:
        model_object.setK(K)
        model = model_object.fit(X)
        preds = model.transform(X)
        y = preds.select(prediction_col).toPandas()[prediction_col]
        results["silouhette"].append(ce.evaluate(preds))
        results["davies_bouldin"].append(davies_bouldin_score(x, y))
        results["calinski_harabasz"].append(calinski_harabasz_score(x, y))
        results["N_K"].append(K)

    if plot_results:
        _, ax = plt.subplots(1, 3, figsize = (15, 5), sharex=True)
        ax[0].plot(results["N_K"], results["silouhette"], label = "Silouhette")
        ax[1].plot(results["N_K"], results["davies_bouldin"], label = "DB")
        ax[2].plot(results["N_K"], results["calinski_harabasz"], label = "CH")
        for ax_ in ax:
            ax_.legend()
            ax_.set_xlabel("N-clusters")
        plt.show()

    return results