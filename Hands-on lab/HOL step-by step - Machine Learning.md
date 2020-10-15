![Microsoft Cloud Workshops](https://github.com/Microsoft/MCW-Template-Cloud-Workshop/raw/master/Media/ms-cloud-workshop.png "Microsoft Cloud Workshops")

<div class="MCWHeader1">
Machine Learning
</div>

<div class="MCWHeader2">
Hands-on lab step-by-step
</div>

<div class="MCWHeader3">
June 2020
</div>

Information in this document, including URL and other Internet Web site references, is subject to change without notice. Unless otherwise noted, the example companies, organizations, products, domain names, e-mail addresses, logos, people, places, and events depicted herein are fictitious, and no association with any real company, organization, product, domain name, e-mail address, logo, person, place or event is intended or should be inferred. Complying with all applicable copyright laws is the responsibility of the user. Without limiting the rights under copyright, no part of this document may be reproduced, stored in or introduced into a retrieval system, or transmitted in any form or by any means (electronic, mechanical, photocopying, recording, or otherwise), or for any purpose, without the express written permission of Microsoft Corporation.

Microsoft may have patents, patent applications, trademarks, copyrights, or other intellectual property rights covering subject matter in this document. Except as expressly provided in any written license agreement from Microsoft, the furnishing of this document does not give you any license to these patents, trademarks, copyrights, or other intellectual property.

The names of manufacturers, products, or URLs are provided for informational purposes only and Microsoft makes no representations and warranties, either expressed, implied, or statutory, regarding these manufacturers or the use of the products with any Microsoft technologies. The inclusion of a manufacturer or product does not imply endorsement of Microsoft of the manufacturer or product. Links may be provided to third party sites. Such sites are not under the control of Microsoft and Microsoft is not responsible for the contents of any linked site or any link contained in a linked site, or any changes or updates to such sites. Microsoft is not responsible for webcasting or any other form of transmission received from any linked site. Microsoft is providing these links to you only as a convenience, and the inclusion of any link does not imply endorsement of Microsoft of the site or the products contained therein.

© 2020 Microsoft Corporation. All rights reserved.

Microsoft and the trademarks listed at <https://www.microsoft.com/en-us/legal/intellectualproperty/Trademarks/Usage/General.aspx> are trademarks of the Microsoft group of companies. All other trademarks are property of their respective owners.

**Contents** 

<!-- TOC -->
- [Machine Learning hands-on lab step-by-step](#machine-learning-hands-on-lab-step-by-step)   
  - [Abstract and learning objectives](#abstract-and-learning-objectives)
  - [Overview](#overview)
  - [Solution architecture](#solution-architecture)
  - [Requirements](#requirements)
  - [Before the hands-on lab](#before-the-hands-on-lab)
  - [Exercise 1: Data exploration and preprocessing](#exercise-1-data-exploration-and-preprocessing)
    - [Task 1: Load, explore and prepare the dataset using a Azure Databricks notebook](#task-1-load-explore-and-prepare-the-dataset-using-a-azure-databricks-notebook)
  - [Exercise 2: Creating a forecast model using automated machine learning](#exercise-2-creating-a-forecast-model-using-automated-machine-learning)
    - [Task 1: Create an automated machine learning experiment](#task-1-create-an-automated-machine-learning-experiment)
    - [Task 2: Review the experiment run results](#task-2-review-the-experiment-run-results)
    - [Task 3: Perform batch inferencing in Azure DataBricks](#task-3-perform-batch-inferencing-in-azure-databricks)
  - [Exercise 3: Using a forecast model for scoring of streaming telemetry](#exercise-3-using-a-forecast-model-for-scoring-of-streaming-telemetry)
    - [Task 1: Create the streaming job using a notebook](#task-1-create-the-streaming-job-using-a-notebook)
  - [Exercise 4: Creating, training and tracking a deep learning text classification model with MLflow and Azure Machine Learning](#exercise-4-creating-training-and-tracking-a-deep-learning-text-classification-model-with-mlflow-and-azure-machine-learning)
    - [Task 1: Create, train and track the classification model using a notebook](#task-1-create-train-and-track-the-classification-model-using-a-notebook)
    - [Task 2: Review model performance metrics and training artifacts in Azure Machine Learning workspace](#task-2-review-model-performance-metrics-and-training-artifacts-in-azure-machine-learning-workspace)
  - [After the hands-on lab](#after-the-hands-on-lab)
    - [Task 1: Clean up lab resources](#task-1-clean-up-lab-resources)

<!-- /TOC -->

# Machine Learning hands-on lab step-by-step

## Abstract and learning objectives

In this hands-on lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy desired models. You will learn how to train a forecasting model against time-series data, without any code, by using automated machine learning, and how to score data in real-time using Spark Structure Streaming within Azure Databricks.  You will create a recurrent neural network (RNN) model using PyTorch in Azure Databricks that can be used to forecast against time-series data and train a Natural Language Processing (NLP) text classification model using Keras. You will also learn how to use MLflow for managing experiments run directly on the Azure Databricks cluster and how MLflow can seamlessly log metrics and training artifacts in your Azure Machine Learning workspace.

At the end of this lab, you will improve your ability to build solutions leveraging Azure Machine Learning and Azure Databricks.

## Overview

Trey Research Inc. delivers innovative solutions for manufacturers. They specialize in identifying and solving problems for manufacturers that can run the range from automating away mundane but time-intensive processes to delivering cutting edge approaches that provide new opportunities for their manufacturing clients.

Trey Research is looking to provide the next generation experience for connected car manufacturers by enabling them to utilize AI to decide when to pro-actively reach out to the customer through alerts delivered directly to the car's in-dash information and entertainment head unit. For their proof of concept (PoC), they would like to focus on two maintenance related scenarios.

In the first scenario, Trey Research recently instituted new regulations defining what parts are compliant or out of compliance. Rather than rely on their technicians to assess compliance, they would like to automatically assess the compliance based on component notes already entered by authorized technicians. Specifically, they are looking to leverage Deep Learning technologies with Natural Language Processing techniques to scan through vehicle specification documents to find compliance issues with new regulations. Then each car is evaluated for out compliance components.

In the second scenario, Trey Research would like to predict the likelihood of battery failure based on the telemetry stream of time series data that the car provides about how the battery performs when the car is started, how it is charging while running and how well it is holding its charge, among other factors. If they detect a battery failure is imminent within the next 30 days, they would like to send an alert.

Upon detection of an out of compliance component or a battery at risk of failure, they would like to be able to send an alert directly to the customer inviting them to schedule a service appointment to replace the part.

In building this PoC, Trey Research wants to understand how they might use machine learning or deep learning in both scenarios, and standardize the platform that would support the data processing, model management and inferencing aspects of each.

They are also interested to learn what new capabilities Azure provides that might help them to integrate with their existing investments in MLflow for managing machine learning experiments. Furthermore, they would also like to understand how Azure might help them to document and explain the models that are created to non-data scientists or might accelerate their time to creating production ready, performant models.

In this lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy the desired models.

## Solution architecture

The following diagram summarizes the key components and processing steps in the lab.

![Vehicle battery telemetry is ingested by an IoT Hub or Event Hub. This data is stored in long term storage, Azure Storage. This data is used by Azure Databricks to train the model that is managed and registered via an Azure Machine Learning workspace. AutoML is also another option that can be used to register a machine learning model. These models are then used for stream data processing and batch data processing in Azure Databricks.](images/lab-architecture.png 'Solution Architecture')

In this lab, models are trained using Azure Machine Learning compute, for automated machine learning using the user experience in the Azure Machine Learning studio and for deep learning with the PyTorch and Keras frameworks in notebooks. Models are registered with the Azure Machine Learning Workspace. The data used for model training is read from Azure Storage.

The scoring is performed using notebooks running within Azure Databricks notebooks, which show how to load and apply the respective models against the data provided.

## Requirements

1. Microsoft Azure subscription must be Pay-As-You-go or MSDN.

    - Trial subscriptions will not work. You will run into issues with Azure resource quota limits.

    - Subscriptions with access limited to a single resource group will not work. You will need the ability to deploy multiple resource groups.

## Before the hands-on lab

Refer to the Before the hands-on lab setup guide manual before continuing to the lab exercises.

## Exercise 1: Data exploration and preprocessing

Duration: 40 minutes

Understanding data through data exploration is one of the core challenges faced today by data engineers and data scientists. Using raw data for modeling can produce misleading results, since data is often noisy and unreliable, and may be missing values. In this exercise, you will explore the raw data, transform and register the  dataset in the Datastore. You will use it to train a forecasting model later in this hands-on-lab. The data preparation steps will be performed on the Azure Databricks cluster.

### Task 1: Load, explore and prepare the dataset using a Azure Databricks notebook

1. Browse to your Azure Databricks Workspace and navigate to AI with Databricks and AML \ 1.0 Data Preparation. This is the notebook you will step through executing in this lab.
2. Follow the instructions within the notebook to complete the lab exercise.

## Exercise 2: Creating a forecast model using automated machine learning

Duration: 40 minutes

In this exercise, you will create a model that predicts battery failure from time-series data using the visual interface to automated machine learning in an Azure Machine Learning workspace.

### Task 1: Create an automated machine learning experiment

1. Navigate to your Azure Machine Learning workspace in the Azure Portal. Select **Try the new Azure Machine Learning studio, Launch now**.

    ![The Azure Machine Learning workspace is displayed. The Launch now button is selected on the Overview screen.](images/automl-open-studio.png 'Open Azure Machine Learning studio')

    > **Note**: Alternatively, you can sign-in directly to the [Azure Machine Learning studio portal](https://ml.azure.com).

2. Select **Automated ML icon** in the left navigation bar.

    ![The Automated ML menu item is highlighted in the left menu in the Azure Machine Learning studio.](images/automl-open.png 'Open Automated ML section')

3. Select **+ New automated ML run**.

    ![In the Automated machine learning section in Azure Machine Learning studio. The "New automated ML run" button is selected.](./images/automl-new-run.png 'Create new automated ML run')

4. Select the `daily-battery-time-series` dataset from the list of registered datasets and then select **Next**. (This dataset was registered as a final step of the previous exercise, from the Azure Databricks notebook.)

     ![In the Create a new Automated ML run dialog, select the daily-battery-time-series dataset from the dataset list. The Next button is highlighted.](images/automl-create-dataset-01.png 'Select registered dataset')

5. Review the dataset details in the `Configure run` section, by selecting the **View dataset** link next to the dataset name.

    ![The Configure run screen shows the option to review the selected dataset structure. Select the view dataset link next to the dataset name.](images/automl-create-dataset-02.png 'Confirm and create the dataset')

6.  Provide the experiment name: `Battery-Cycles` and select **Daily_Cycles_Used** as target column. Select **Create a new compute**.

    ![In the Configure run form is populated with the above values. The Create a new compute button is highlighted.](images/automl-create-experiment.png 'Create New Experiment details')

7.  For the new compute, provide the following values and then select **Create**:

    - **Compute name**: `auto-ml-compute`
  
    - **Select Virtual Machine size**: `STANDARD_DS3_v2`
  
    - **Minimum number of nodes**: `1`
  
    - **Maximum number of nodes**: `1`

    ![The New Training Cluster form is populated with the above values. The Create button is selected at the bottom of the form.](images/automl-create-compute.png 'Create a New Compute')

    > **Note**: The creation of the new compute may take several minutes. Once the process is completed, select **Next** in the `Configure run` section.

9.  Select the `Time series forecasting` task type and provide the following values and then select **View additional configuration settings**:

    - **Time column**: `Date`

    - **Time series identifier(s)**: `Battery_ID`

    - **Forecast horizon**: `30`

    ![The Select task type form is populated with the values outlined above. The View additional configuration settings link is highlighted.](images/automl-configure-task-01.png 'Configure time series forecasting task')

10. For the automated machine learning run additional configurations, provide the following values and then select **Save**:

    - **Primary metric**: `Normalized root mean squared error`

    - **Training job time (hours)** (in the `Exit criterion` section): enter `1` as this is the lowest value currently accepted.

    - **Metric score threshold**: enter `0.1355`. When this threshold value will be reached for an iteration metric the training job will terminate.

    ![The Additional configurations form is populated with the values defined above. The Save button is highlighted at the bottom of the form.](images/automl-configure-task-02.png 'Configure automated machine learning run additional configurations')

    > **Note**: We are setting a metric score threshold to limit the training time. In practice, for initial experiments, you will typically only set the training job time to allow AutoML to discover the best algorithm to use for your specific data.

11. Select **Finish** to start the new automated machine learning run.

    > **Note**: The experiment should run for up to 10 minutes. If the run time exceeds 15 minutes, cancel the run and start a new one (steps 3, 9, 10). Make sure you provide a higher value for `Metric score threshold` in step 10.

### Task 2: Review the experiment run results

1. Once the experiment completes, select `Details` to examine the details of the run containing information about the best model and the run summary.

   ![The Run Detail screen of Run 1 indicates it has completed. The Details tab is selected where the the best model, ProphetModel, is indicated along with the run summary.](images/automl-review-run-01.png 'Run details - best model and summary')

2. Select `Models` to see a table view of different iterations and the `Normalized root mean squared error` score for each iteration. Note that the normalized root mean square error measures the error between the predicted value and actual value. In this case, the model with the lowest normalized root mean square error is the best model. Note that Azure Machine Learning Python SDK updates over time and gives you the best performing model at the time you run the experiment. Thus, it is possible that the best model you observe can be different than the one shown below.

    ![The Run Detail screen of Run 1 is displayed with the Models tab selected. A table of algorithms is displayed with the values for Normalized root mean squared error highlighted.](images/automl-review-run-02.png 'Run Details - Models with their associated primary metric values')

3. Return to the details of your experiment run and select the best model **Algorithm name**.

    ![The Run Detail screen of Run 1 is displayed with the Details tab selected. The best model algorithm name is selected.](images/automl-review-run-03.png 'Run details - recommended model and summary')

4. From the `Model` tab, select **View all other metrics** to review the various `Run Metrics` to evaluate the model performance.

    ![The model details page displays run metrics associated with the Run.](images/automl-review-run-04.png 'Model details - Run Metrics')

5. Next, select **Metrics, predicted_true** to review the model performance curve: `Predicted vs True`.

    ![The model run page is shown with the Metrics tab selected. A chart is displayed showing the Predicted vs True curve.](images/automl-review-run-05.png 'Predicted vs True curve')

### Task 3: Perform batch inferencing in Azure DataBricks




## Exercise 3: Using a forecast model for scoring of streaming telemetry

Duration: 45 minutes

In this exercise, you will apply the forecast model to a Spark streaming job in order to make predictions against streaming data.

### Task 1: Create the streaming job using a notebook

1. Browse to your Azure Databricks Workspace and navigate to `AI with Databricks and AML \ 2.0 Stream Scoring`. This is the notebook you will step through executing in this lab.

2. Follow the instructions within the notebook to complete the lab.

## Exercise 4: Creating, training and tracking a deep learning text classification model with MLflow and Azure Machine Learning

Duration: 45 minutes

In this exercise, you create a model for classifying component text as compliant or non-compliant. You will train the model Azure Machine Learning and use MLflow integration with Azure Machine Learning to track and log experiment metrics and artifacts in the Azure Machine Learning workspace.

### Task 1: Create, train and track the classification model using a notebook

1. Browse to your Azure Databricks Workspace and navigate to `AI with Databricks and AML \ 3.0 Deep Learning with Text`. This is the notebook you will step through executing in this lab.

2. Follow the instructions within the notebook to complete the lab.

### Task 2: Review model performance metrics and training artifacts in Azure Machine Learning workspace

1. Select the **Link to Azure Machine Learning studio** from the output of the last cell in the notebook to open the `Run Details` page in the Azure Machine Learning studio.

   ![A notebook cell output is displayed with the Link to Azure Machine Learning studio highlighted under the Details Page column.](images/mlflow_1.png 'Open Azure Machine Learning studio')

2. The **Run Details** page shows the three metrics that were logged via MLflow during the model training process: **learning rate (lr)**, **evaluation loss (eval_loss)**, and **evaluation accuracy (eval_accuracy)**.

   ![In the Run Details page, the Metrics section containing eval_accuracy, eval_loss, and lr is highlighted.](images/mlflow_2.png 'Model Training Metrics')

3. Next, select **Outputs + logs, training_results.png** to review the model training artifacts logged using MLflow. In this section, you can review the curves showing both accuracy and loss as the model training progress. You can also observe that MLflow logs the trained model and the training history with Azure Machine Learning workspace.

   ![On the Run Details page, the Output + Logs tab is selected, and the training_results.png item is selected in a list on the left. The image is displayed showing charts of Training and validation accuracy, and Training and validation loss.](images/mlflow_3.png 'Model Training Artifacts')

## After the hands-on lab

Duration: 5 minutes

To avoid unexpected charges, it is recommended that you clean up all of your lab resources when you complete the lab.

### Task 1: Clean up lab resources

1. Navigate to the Azure Portal and locate the `MCW-Machine-Learning` Resource Group you created for this lab.

2. Select **Delete resource group** from the command bar.

    ![The Delete resource group button.](images/cleanup-delete-resource-group.png 'Delete resource group button')

3. In the confirmation dialog that appears, enter the name of the resource group and select **Delete**.

4. Wait for the confirmation that the Resource Group has been successfully deleted. If you don't wait, and the delete fails for some reason, you may be left with resources running that were not expected. You can monitor using the Notifications dialog, which is accessible from the Alarm icon.

    ![The Notifications dialog box has a message stating that the resource group is being deleted.](images/cleanup-delete-resource-group-notification-01.png 'Notifications dialog box')

5. When the Notification indicates success, the cleanup is complete.

    ![The Notifications dialog box has a message stating that the resource group has been deleted.](images/cleanup-delete-resource-group-notification-02.png 'Notifications dialog box')

You should follow all steps provided _after_ attending the Hands-on lab.
