from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler, MinMaxScaler, StringIndexer, PCA
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("ML").master("local[*]").getOrCreate()
spark.conf.set("spark.sql.ansi.enabled", False)
data_dir = "../dataset/e-shop clothing 2008.csv"

eu_codes = [2, 3, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 30, 34, 35, 36, 37, 41]
non_eu_europe_codes = [7, 19, 28, 31, 32, 33, 38, 39]
outside_europe_codes = [1, 4, 5, 6, 20, 26, 40, 42]
unidentified_codes = [12, 43, 44, 45, 46, 47]


def load_data(dir: str):
    return spark.read.csv(dir, header=True, inferSchema=True, sep = ";")

def tokenize(raw_df: DataFrame, build_sequences = True, order_col = "order", grouping_col = "session ID",  to_ignore: list[str] = ["year", "month", "day", "country"]):
    '''
    tokenizes clicks ingoring the mentioned colums, by default ignores "year", "month", "day", "country"\n
    returns tokens and original dataframe with "id" column
    '''
    token_features = [col for col in raw_df.columns if col not in to_ignore + [order_col, grouping_col]]
    tokens = raw_df.select(*token_features).drop_duplicates()
    tokens = tokens.withColumn("id", F.monotonically_increasing_id())
    tokenized = raw_df.join(tokens, on = token_features, how="left").drop(*token_features)
    if build_sequences:
        w = Window.partitionBy(grouping_col).orderBy(order_col)
        tokenized = (tokenized
            .withColumn("sequence", F.collect_list("id").over(w))
            .withColumn("rn", F.row_number().over(w.orderBy(F.desc(order_col)))).filter(F.col("rn") == 1)
            .orderBy(grouping_col).drop("rn", "id"))
    print("TOKENIZED DF:")
    tokenized.show(2)
    print("TOKEN INFO: ")
    tokens.show(2)
    return tokenized, tokens

def fit_zipfs(freqs: DataFrame, B = 10, frequency_col = "count", id_col = "id", ignore_bottom = 0, ignore_top=0, plot = True):
    '''
    Fits and evaluates Zipf's law and visualizes distributions\n
    Set  ignore_bottom = 0, ignore_top=0 to ignore in fitting and evaluation\n
    '''

    def zipfs_f(ranks: DataFrame, C, alpha):
        preds = ranks.withColumn("density_pred", C / (F.col("rank") + F.lit(B))**alpha)
        return preds
    
    freqz = freqs.orderBy(frequency_col, ascending = False)
    size = freqz.count()
    freqz = freqz.withColumn("rank", F.monotonically_increasing_id() + 1)
    freqz = freqz.withColumn("log_ranks", F.log(F.col("rank") + F.lit(B)))
    freqz = freqz.withColumn("log_freqs", F.log(frequency_col))

    #linear log-log fit
    va_ranks = VectorAssembler(inputCols=["log_ranks"], outputCol="log_ranks_v")
    reg_zipf = LinearRegression(featuresCol="log_ranks_v", labelCol="log_freqs", predictionCol="estimate", regParam=1e-8)
    
    filtered = freqz
    if ignore_bottom != 0 or ignore_top != 0:
        print("Ignored %d top and %d bottom tokens for estimation and fitting" %(ignore_top, ignore_bottom))
        filtered = freqz.filter((F.col('rank') < (size - ignore_bottom)) & (F.col('rank') > ignore_top)).withColumn("rank", F.monotonically_increasing_id() + 1)

    zipfs_pipeline = Pipeline(stages = [va_ranks, reg_zipf]).fit(filtered)
    zipfs_model = zipfs_pipeline.stages[-1]

    coefs = zipfs_model.coefficients.toArray()
    log_C = zipfs_model.intercept
    C = np.exp(log_C)
    alpha = -coefs[0] 

    preds = zipfs_pipeline.transform(freqz).select(id_col, "rank", "log_ranks", "log_freqs", frequency_col, "estimate")
    final = zipfs_f(preds, C, alpha=alpha)

    if plot:
        preds_pd = final.toPandas()
        plt.figure(figsize=(10, 6))
        plt.plot(preds_pd["log_ranks"], preds_pd["log_freqs"])
        plt.plot(preds_pd["log_ranks"], preds_pd["estimate"])
        plt.title("Zipf's log-log curve")
        plt.ylabel("Log Token frequency")
        plt.xlabel("Log Token rank + B")
        plt.show()
        print("Zipf's fit R-squared: ", zipfs_model.summary.r2)
        print("Zipf's fit MSE: ", zipfs_model.summary.meanSquaredError)
        print("Zipf's fit Explained Variance: ", zipfs_model.summary.explainedVariance)
        print(f"C estimate: {C:.2f}")
        print(f"Alpha estimate: {alpha:.2f}")

        preds_pd["residual"] = preds_pd["log_freqs"] - preds_pd["estimate"]

        plt.figure(figsize=(10, 4))
        plt.plot(preds_pd["rank"], preds_pd["residual"])
        plt.axhline(0, linestyle="--")
        plt.title("Residuals vs Rank")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(preds_pd["rank"], preds_pd[frequency_col], label = "Original")
        plt.title("Original density curve vs Zipf's estimate")
        plt.plot(preds_pd["rank"], preds_pd["density_pred"], label = "Estimate")
        plt.legend()
        plt.ylim((0,preds_pd[frequency_col].max()))
        plt.ylabel("Frequency")
        plt.xlabel("Token Rank")
        plt.show()
        return final, preds_pd
    return final


def prepare_data_pipeline(df: DataFrame, create_features_only = False, omit: list[str] = ["day", "month", "year", "order", "session ID"]):
    '''
    Returns Dataframe and pipeline objects\n
    Splits the "page 2 (clothing model)" into model_letter and model_number\n
    Groups countries by region
    Minmax day (currently drops it)\n
    One-hots categorical\n
    Omits year, month, order, id by default
    PCA is the last stage\n
    '''
    # add region column based on country codes
    _df = df.withColumn("country", F.when(F.col("country") == 29, 0)
        .when(F.col("country").isin(eu_codes), 1)
        .when(F.col("country").isin(non_eu_europe_codes), 2)
        .when(F.col("country").isin(outside_europe_codes), 3)
        .otherwise(4)).withColumn("index", F.monotonically_increasing_id())
                                                                              
    #split clothing model strings into letter and number
    string_col = "page 2 (clothing model)"
    _df = _df.withColumn("model_letter", F.substring(string_col, 1, 1)).withColumn("model_number", F.substring(string_col, 2, 3)).drop(string_col)

    '''
    # period of 31 also tested, no benefit
    _df = (
        _df
        .withColumn("date",F.to_date(F.concat_ws("-", F.col("year"), F.col("month"), F.col("day")), "yyyy-M-d"))
        .drop("month", "year", "day")
        .withColumn("month", F.month("date"))
        .withColumn("weekday", F.dayofweek("date") - 1)   # 0..6
        .drop("date"))
    _df = (
        _df
        .withColumn("day_sin", F.sin(F.col("weekday") / 7 * 2 * np.pi))
        .withColumn("day_cos", F.cos(F.col("weekday") / 7 * 2 * np.pi))
        .drop("weekday"))
    '''
    
    _df = _df.withColumnRenamed("page 1 (main category)", "main_category")
    cols = [col for col in _df.columns if col not in omit]
    cols.remove("index")

    if not create_features_only:
        #prepare column names
        numeric_cols = ["price"]
        string_cols = ["model_letter", "model_number"]
        binary_cols = ["model photography"]

        si_output = [col + "_id" for col in string_cols]
        binary_output = [col + "_id" for col in binary_cols]

        categorical_cols = [col for col in cols if col not in numeric_cols + string_cols + binary_cols] + si_output
        final_cols = [col+ "_enc"  for col in categorical_cols] + [col + "_s" for col in numeric_cols] + binary_output

        #first encode letters, and additionally convert binary to 0, 1 index (1,2 now)
        si = StringIndexer(inputCols=string_cols + binary_cols, outputCols=si_output + binary_output)

        #onehot categorical columns
        ohe = OneHotEncoder(inputCols=categorical_cols, outputCols=[col + "_enc" for col in categorical_cols])

        #va's for continous, vectorize continous
        va_1 = VectorAssembler(inputCols=["price"], outputCol="price_v")
       #va_2 = VectorAssembler(inputCols=["day_cos"], outputCol="day_cos_s")
       #va_3 = VectorAssembler(inputCols=["day_sin"], outputCol="day_sin_s")
        vas = [va_1]#, va_2]

        # scalers 
        ss = StandardScaler(inputCol = "price_v", outputCol="price_s")
       # mm = MinMaxScaler(inputCol= "day_v", outputCol="day_s") 

        #Final vector 
        va_final = VectorAssembler(inputCols=final_cols, outputCol="vector")

        #Reduce
        pca = PCA(k = 30, inputCol="vector", outputCol="reduced")

        stages = [*vas, si, ohe, ss, va_final, pca] #mm,
        pipe = Pipeline(stages = stages)
        _df.show(5)
        return _df, pipe
    return _df

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