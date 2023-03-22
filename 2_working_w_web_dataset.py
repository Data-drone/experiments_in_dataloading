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
import numpy as np
import glob

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

user_path = f'/Users/{username}/data/imagenette_wds'
imagenette_tar_data_path = f'/dbfs{user_path}'
dbfs_path = f'dbfs:{user_path}'

# COMMAND ----------

# web dataset reader in dali needs the full list of files

tar_path = os.path.join(imagenette_tar_data_path, '*.tar')

tar_list = glob.glob(tar_path)

# COMMAND ----------

# MAGIC %md
# the webdataset reader is a bit complex.
# paths needs to be the exact list of files

# ext will be used to find the subfiles in the tar
# in the case of our imagnette data the image and json data were stored as
# filename (including .JPEG) followed by .image and .json
# using tar -xvf on one of the sample files is the easiest way to check this

# Note that errors when they occur will be listed as:
# num_shard errors which can be confusing

# COMMAND ----------


@pipeline_def(batch_size=32, num_threads=4, device_id=0)
def wds_pipeline(wds_data=tar_list):
    img_raw, json = fn.readers.webdataset(
        paths=wds_data,
        ext=["JPEG.image", "JPEG.json"],
        num_shards=1,
        missing_component_behavior="skip"
    )
        
#     img = fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)
#     resized = fn.resize(img, device="gpu", resize_shorter=256.)
#     output = fn.crop_mirror_normalize(
#         resized,
#         dtype=types.FLOAT,
#         crop=(224, 224),
#         mean=[0., 0., 0.],
#         std=[1., 1., 1.])
    return img_raw, json

# COMMAND ----------

# get pipeline ready for use
pipe = wds_pipeline()
pipe.build()

# COMMAND ----------

# test that the pipeline works
for i in range(10): 
  pipe_out = pipe.run()
  print(pipe_out)

# COMMAND ----------

