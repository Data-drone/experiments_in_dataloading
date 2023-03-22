# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Making Delta Dataset from images
# MAGIC We will start by making a Delta dataset
# MAGIC we will test out some of these and see how it performs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup

# COMMAND ----------

import os
from pyspark.sql import functions as F

# COMMAND ----------

# Image dataset
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

user_path = f'/Users/{username}/data/imagenette2'
imagenette_data_path = f'/dbfs{user_path}'
dbfs_path = f'dbfs:{user_path}'

train_subpath = 'train'
val_subpath = 'val'

# COMMAND ----------


# Lets read it in with spark then process it as a dataframe
dbutils.fs.ls(os.path.join(dbfs_path, train_subpath))

# COMMAND ----------

def process_folder(file_subpath, data_split):

    processed_df = spark.read.format("image") \
        .option("recursiveFileLookup","true") \
        .load(os.path.join(dbfs_path, file_subpath, '*', '*.JPEG'))
    
    # string parsing and data extraction
    processed_df = processed_df \
                    .withColumn('split', F.lit(data_split)) \
                    .withColumn('height', F.col('image.height')) \
                    .withColumn('width', F.col('image.width')) \
                    .withColumn('nChannels', F.col('image.nChannels')) \
                    .withColumn('origin', F.col('image.origin')) \
                    .withColumn('filename', F.element_at(F.split(F.col('image.origin'), '/'), -1))
    
    processed_df = processed_df \
                    .withColumn('target_category', F.element_at(F.split(F.col('image.origin'), '/'), -2))

    
    return processed_df

# COMMAND ----------

# we need to use recursive file lookup to handle the nested structures
test_frame = process_folder(train_subpath, train_subpath)
val_frame = process_folder(val_subpath, val_subpath)

merged_dataset = test_frame.union(val_frame)

# COMMAND ----------
%sql

CREATE DATABASE IF NOT EXISTS imagenette_delta

# COMMAND ----------


# Save out the dataset

merged_dataset.write.mode('overwrite').saveAsTable('imagenette_delta.train_val_set')

# COMMAND ----------