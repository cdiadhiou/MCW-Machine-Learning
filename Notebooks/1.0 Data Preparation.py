# Databricks notebook source
# MAGIC %md # Cleaning and Preparing Data
# MAGIC In this notebook, you will preprocess the car battery raw dataset and store it in Azure Machine Learning.

# COMMAND ----------

import uuid
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from azureml.core import Workspace, Dataset

# upload the local file to a datastore on the cloud
subscription_id = '<your-subscription-id>'
resource_group = 'MCW-Machine-Learning'
workspace_name = 'mcwmachinelearning'

filename='daily-battery-time-series-v3.csv'
filename_processed = 'daily-battery-time-series-v3-processed.csv'

# COMMAND ----------

# MAGIC %md ### Download the data
# MAGIC 
# MAGIC The following cell will download the data set containing the daily battery time series.
# MAGIC The CSV file will be saved in a temporary folder on the Databricks cluster for later reuse.

# COMMAND ----------

# Create a temporary folder to store locally relevant content for this notebook
tempFolderName = '/FileStore/CarBatteries_{0}'.format(uuid.uuid4())
dbutils.fs.mkdirs(tempFolderName)
print('Content files will be saved to {0}'.format(tempFolderName))

downloadCommand = 'wget -O ''/dbfs{0}/{1}'' ''https://databricksdemostore.blob.core.windows.net/data/connected-car/{1}'''.format(tempFolderName, filename)
ret_val = os.system(downloadCommand)

print('System returned value %s for download command.' % (ret_val))

# COMMAND ----------

# MAGIC %md Run a simple check to make sure our data file was correctly downloaded.

# COMMAND ----------

# Check that all files are successfully downloaded
dbutils.fs.ls(tempFolderName)

# COMMAND ----------

# MAGIC %md ### Load the data
# MAGIC 
# MAGIC The previously downloaded CSV file will be loaded into a Pandas Dataframe and inspected.

# COMMAND ----------

# Load the dataset
file_path = '/dbfs%s/%s' % (tempFolderName, filename)
df = pd.read_csv(file_path, delimiter=',', index_col='Date', parse_dates=['Date'])

# Inspect the data frame
df.head()

# COMMAND ----------

# Remove the temporary folder
dbutils.fs.rm(tempFolderName, recurse=True)

# COMMAND ----------

# MAGIC %md ### Analyze the data
# MAGIC 
# MAGIC We notice that initially there seem to be entries every twelve hours, whereas later the entries are every twenty four hours, we'll plot the daily cycles to get a better idea of how the data looks like.

# COMMAND ----------

df['Daily_Cycles_Used'].plot(figsize=(16,6))

# COMMAND ----------

print(f'Entries date starts at {df.index[0]}')
print(f'Entries date ends at {df.index[-1]}')
print(f'Dataset spans {(df.index[-1] - df.index[0]).days + 1} days but has {df.shape[0]} entries')

# COMMAND ----------

# MAGIC %md Check how many 12-hour entries are

# COMMAND ----------

df[df.index.hour == 12]

# COMMAND ----------

# MAGIC %md Check how the 12-hour entries are distributed

# COMMAND ----------

# Count the entries for each date
counts = df.groupby(df.index.date).count()

# Display the value distribution for one of the columns
counts['Battery_ID'].plot.area(figsize=(16,6))

# COMMAND ----------

# MAGIC %md ### Conclusions
# MAGIC 
# MAGIC We notice the following:
# MAGIC * The entries are recorded twice a day up until the end of 2014, and once a day starting from 2015
# MAGIC * There is a significant number of possible duplicate entries
# MAGIC 
# MAGIC ### Make sure that our conclusions are correct

# COMMAND ----------

# Check that indeed the frequency change starts in January 2015
df['2014-12-25':'2015-1-10']

# COMMAND ----------

df.index

# COMMAND ----------

# Check for duplicate entries in the time series
df.index[df.index.duplicated()]

# COMMAND ----------

# MAGIC %md ### Preprocess data
# MAGIC 
# MAGIC First, drop the duplicate entries

# COMMAND ----------

df = df.drop(df.index[df.index.duplicated()])

# COMMAND ----------

# Check that this fixes the counts chart
counts = df.groupby(df.index.date).count()
counts['Battery_ID'].plot.area(figsize=(16,6))

# COMMAND ----------

# MAGIC %md Next, we need to update the entries' frequency.

# COMMAND ----------

# Split the initial dataset into two smaller ones, each with their own frequency
df_12h = df[:'2014']
df_24h = df['2015':]
df_12h

# COMMAND ----------

# MAGIC %md Time series data might contain missing entries, so check for that too

# COMMAND ----------

# Check how many entries the dataset would have if it had entries every twelve hours - pandas will create empty entries if none exist, so this is a good way to check for missing values
df_12h_all = df_12h.asfreq('12h')
df_12h_all

# COMMAND ----------

print(f'Raw dataset has {df_12h.shape[0]} entries while it should have had {df_12h_all.shape[0]} entries.')

# COMMAND ----------

# Check statistics
df_12h_all.info()

# COMMAND ----------

# Display missing entries
df_12h_all[df_12h_all.isnull().any(axis=1)]

# COMMAND ----------

# Use time interpolation to fill in missing entries
df_12h = df_12h_all.interpolate(method='time')
df_12h

# COMMAND ----------

# Check statistics again
df_12h.info()

# COMMAND ----------

# MAGIC %md In order to change the entries' frequency we need to resample the data and aggregate it. We also have different concerns for aggregating depending on the features.

# COMMAND ----------

df_12h = df_12h.resample('1D').agg({
    'Battery_ID': np.min, # always 0
    'Battery_Age_Days': np.min, # pick the minimum value, since the 12-hour values are intermediary
    'Number_Of_Trips': np.sum,
    'Daily_Trip_Duration': np.sum,
    'Daily_Cycles_Used': np.sum,
    'Lifetime_Cycles_Used': np.sum,
    'Battery_Rated_Cycles': np.min  # always 200
})

# Lifetime_Cycles_Used aggregates the Daily_Cycles_Used for the current date, so update accordingly
df_12h['Lifetime_Cycles_Used'] = df_12h['Daily_Cycles_Used'].expanding().sum()
df_12h

# COMMAND ----------

# MAGIC %md Check if the second dataset has missing entries - notice that we use a different frequency for this

# COMMAND ----------

df_24h.asfreq('1D').info()

# COMMAND ----------

# Interpolate the second dataset, too
df_24h = df_24h.asfreq('1D').interpolate(method='time')
df_24h

# COMMAND ----------

# Concatenate the two datasets
df = pd.concat([df_12h, df_24h])
df

# COMMAND ----------

# Make sure the data looks fine
df['Daily_Cycles_Used'].plot(figsize=(16,6))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Persist the processed dataset
# MAGIC Save the dataset to Azure Storage

# COMMAND ----------

# Connect to the Azure ML Workspace
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Get default datastore to upload prepared data
datastore = workspace.get_default_datastore()

# COMMAND ----------


# Create a temporary folder to store locally relevant content for this notebook
tempFolderName = '/FileStore/CarBatteries_{0}'.format(uuid.uuid4())
dbutils.fs.mkdirs(tempFolderName)

file_path = '/dbfs%s/%s' % (tempFolderName, filename_processed)

# Persist dataset to temporary folder
df.to_csv(file_path, sep=',')

# datastore.upload(src_dir=, target_path='data')
datastore.upload_files([file_path], relative_root=None, target_path=None, overwrite=True, show_progress=True)
dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, (filename_processed))])
dataset = dataset.register(workspace=workspace,
                                 name='daily-battery-time-series',
                                 description='cat baterry time series processed dataset',
                                create_new_version=True)
# Remove temporary folder
dbutils.fs.rm(tempFolderName, recurse=True)
