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

%pip install git+https://github.com/NVIDIA-Merlin/NVTabular.git@v23.02.00

# COMMAND ----------

%pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com

# COMMAND ----------

%pip install protobuf==3.20.0


# COMMAND ----------

import nvtabular as nvt
import glob

# COMMAND ----------

test_table_path = '/dbfs/user/hive/warehouse/brian_rossman.db/train_data/'

extensions = '*.parquet'


# COMMAND ----------


# An existing test file

# we need to find a parquet file
# we shouldn't do this and should use a manifest file instead since this is a delta table
dataset = glob.glob(extensions)


# COMMAND ----------

dataset = nvt.Dataset(dataset, engine="parquet")

# COMMAND ----------