# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Creating a web dataset
# MAGIC we will create a webdataset

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup


# COMMAND ----------

import os
import webdataset as wds
from pyspark.sql import functions as F
import json

import io
import PIL
from PIL import Image

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

user_path = f'/Users/{username}/data/imagenette_wds'
imagenette_tar_data_path = f'/dbfs{user_path}'
dbfs_path = f'dbfs:{user_path}'

# clear the dir
dbutils.fs.rm(dbfs_path, True)

dbutils.fs.mkdirs(dbfs_path)

# COMMAND ----------


# MAGIC %md
# MAGIC Webdataset requires that the data be formated in a certain way.
# MAGIC
# MAGIC Data is fed into the python writer via as a dict. Which
# MAGIC needs a key called key:  __key__ this is usually a primary key or file name
# MAGIC 
# MAGIC The fields of the dict by default need to match with the handlers specified.
# MAGIC It is possible to override the handler but we will not explore that at this stage.
# MAGIC
# MAGIC For default Handlers see: https://github.com/webdataset/webdataset/blob/039d74319ae55e5696dcef89829be9671802cf70/webdataset/writer.py#L150
# MAGIC So valid dictionary keys in your dict are: 'txt', 'html' 'pickle', 'json' amongst others
# MAGIC We will encode metadata into the json format and the image as PIL 

# COMMAND ----------

# TODO - We should try to use the binary from the delta file and not
# pull the raw

# define a function to save a partition as a WebDataset
def save_partition_webdataset(iter, output_path):
    
    partition_string = str(uuid.uuid4())
    
    file_output_path = os.path.join(output_path, f'imagenette_{partition_string}.tar')
  
    with wds.TarWriter(file_output_path) as writer:
        for row in iter:
            # convert the row to a dictionary and write it to the tar file
            row_dict = row.asDict()
            
            # we need to assume a format here
            parsed_dict = {}
            parsed_dict['__key__'] = row_dict['filename']
            
            # io.BytesIO(row_dict['image']['data'])
            # we need to switch this
            dbfs_dir = row_dict['image']['origin']
            filepathing = re.sub(r'^dbfs:', '/dbfs', dbfs_dir)
            
            parsed_dict['image'] = Image.open(filepathing)
            
            ## now we will make a json
            row_dict.pop('filename')
            row_dict.pop('image')
            
            parsed_dict['json'] = json.dumps(row_dict)
            
            writer.write(parsed_dict)

# COMMAND ----------


# define the output path for the WebDataset
output_path = os.path.join(imagenette_tar_data_path, 'imagenette.tar')

# load the Spark table
df = spark.read.table("imagenette_delta.train_val_set")

# COMMAND ----------


# save the Spark table as a WebDataset in parallel
df.foreachPartition(lambda iter: save_partition_webdataset(iter, output_path))

