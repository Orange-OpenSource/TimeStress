# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import json
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.linalg import SparseVector
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionDescription, VectorParams


from _old_codebase.utils.wd import ent_to_vec
from _old_codebase.config import MONGO_URL
import os


# the proxy format must be : http[s]://URL:PORT
def get_java_extra_options():
    def preprocess_env(env):
        proxy = os.getenv(env)
        if len(proxy) == 0:
            return "", ""
        if proxy.startswith("https://"):
            proxy = proxy[8:]
        if proxy.startswith("http://"):
            proxy = proxy[7:]
        return proxy.rsplit(":", 1)

    http_host, http_port = preprocess_env("http_proxy")
    https_host, https_port = preprocess_env("https_proxy")
    options = ""
    if len(http_host):
        options += f"-Dhttp.proxyHost={http_host} -Dhttp.proxyPort={http_port} "
    if len(https_host):
        options += f"-Dhttps.proxyHost={https_host} -Dhttps.proxyPort={https_port} "
    options += "-Dio.netty.tryReflectionSetAccessible=true"
    return options


java_extra_options = get_java_extra_options()


def process_one_entity(ent_id, claims):
    ent_vec = ent_to_vec(claims, add_singles=True)
    return ent_vec


# Create a Spark session
spark = (
    SparkSession.builder.appName("MongoDBToQdrant")
    .config("spark.mongodb.input.uri", f"{MONGO_URL}/wiki")
    .config("spark.mongodb.output.uri", f"{MONGO_URL}/wiki")
    .config(
        "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1"
    )
    .config("spark.driver.memory", "15g")
)
if len(java_extra_options):
    spark.config("spark.driver.extraJavaOptions", java_extra_options)
spark = spark.getOrCreate()
# Read the MongoDB collection into a Spark DataFrame
pipeline = json.dumps(
    {
        "$project": {
            "claims": 1,
        }
    }
)

df = (
    spark.read.format("mongodb")
    .option("spark.mongodb.read.database", "wiki")
    .option("spark.mongodb.read.collection", "wikidata_new_json")
    .option("spark.mongodb.read.connection.uri", MONGO_URL)
    .option("aggregation.pipeline", pipeline)
    .load()
    .limit(10)
)
# Register the elt2txt function as a UDF
elt2txt_udf = udf(process_one_entity, ArrayType(StringType()))
df_with_all_columns = df.withColumn(
    "all_columns", struct([df[col] for col in df.columns])
)
df_with_words = df_with_all_columns.withColumn("words", elt2txt_udf("all_columns"))

# Compute the TF-IDF vectors
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1024)
featurizedData = hashingTF.transform(df_with_words)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Select the features vector for further processing
tfidf_vectors = rescaledData.select("features")


# Initialize the Qdrant client
qdrant_client = QdrantClient(host="127.0.0.1", port=6334)

# Define the name of the collection you want to create
collection_name = "Entity"

# Define the vector size (number of dimensions)
vector_size = tfidf_vectors.shape[1]  # Replace with the actual size of your vectors

# Check if the collection exists
collections = qdrant_client.collections.list()
existing_collections = [col.name for col in collections.result.collections]

if collection_name not in existing_collections:
    # Define the collection parameters
    collection_description = CollectionDescription(
        vector_size=vector_size,
        distance="Cosine",  # You can choose other distances like Euclidean, Dot, etc.
        vector_params=VectorParams(size=vector_size, distance="Cosine"),
        # If you have other fields to index, define them here
        # field_index_operations=[
        #     FieldIndexOperations(
        #         field_name="your_field_name",
        #         operation="Create",
        #         field_index_params=FieldIndexParams(
        #             index_type="Plain",  # You can choose other index types like Hnsw, etc.
        #             field_type="Int"  # Choose the appropriate field type (Int, Float, Keyword, etc.)
        #         )
        #     )
        # ]
    )

    # Create the collection
    qdrant_client.collections.create(
        collection_name=collection_name, collection_description=collection_description
    )
else:
    qdrant_client.collections.delete(collection_name=collection_name)

# Push the vectors to Qdrant
for index, row in tfidf_vectors.collect():
    # Ensure the vector is a SparseVector
    if not isinstance(row["features"], SparseVector):
        continue

    # Convert the SparseVector to a dictionary format
    vector = {
        int(i): float(x)
        for i, x in zip(row["features"].indices, row["features"].values)
    }

    # Create the Qdrant point
    point = PointStruct(payload={}, vector=vector)

    # Add the point to Qdrant
    qdrant_client.points.replace(
        point_id=index, points=[point], collection_name="your_collection_name"
    )

# Stop the Spark session
spark.stop()
