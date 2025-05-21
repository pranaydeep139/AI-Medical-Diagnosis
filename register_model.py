from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os

# --- Configuration ---
# Replace with your Azure subscription ID, resource group, and workspace name
subscription_id = "272bbf4c-ec2f-4afa-b0b9-2b3b62483225"
resource_group = "rg-pranaydeep139-5281_ai"
workspace_name = "medazure"

model_name = "diagnosis-assistant" # Choose a name for your model in Azure ML
model_description = "Random Forest model to predict diseases based on symptoms, with supporting JSON artifacts."
local_model_path = "./aml_model_files/" # Path to the FOLDER containing your model files

# --- Get MLClient ---
try:
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )
    print(f"Successfully connected to workspace: {ml_client.workspace_name}")
except Exception as e:
    print(f"Error connecting to Azure ML workspace: {e}")
    exit()

# --- Define the Model ---
# The 'path' argument points to your local folder. Azure ML will upload its contents.
model_to_register = Model(
    name=model_name,
    description=model_description,
    path=local_model_path, # This is the crucial part!
    type=AssetTypes.CUSTOM_MODEL # Or MLFLOW_MODEL if it were an MLflow model, etc.
                                  # CUSTOM_MODEL is a good generic choice for scikit-learn + custom files.
    # version="1" # Optional: Azure ML will auto-increment the version if not specified
    # tags={"framework": "scikit-learn", "task": "classification"} # Optional
)

# --- Register the Model ---
try:
    registered_model = ml_client.models.create_or_update(model_to_register)
    print(f"Model '{registered_model.name}' version '{registered_model.version}' registered successfully.")
    print(f"View in Azure ML Studio: {registered_model.id}")
except Exception as e:
    print(f"Error registering model: {e}")