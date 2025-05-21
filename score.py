import json
import os
import joblib
import pandas as pd
import numpy as np
import logging
import traceback

# Global variables to hold the model and associated artifacts
model = None
feature_order = None
disease_names = None
logger = None

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory,
    storing a static global variable, etc.
    """
    global model, feature_order, disease_names, logger

    # Initialize logger
    # The AML environment automatically configures a logger named "azureml.score",
    # which outputs to AzureML logs.
    logger = logging.getLogger("azureml.score")
    logger.info("In init() method. Initializing scoring script.")

    # Get the path to the deployed model directory. AZUREML_MODEL_DIR is an environment variable
    # that points to the directory where model files are deployed.
    model_root = os.getenv("AZUREML_MODEL_DIR")

    # Fallback for local testing (if AZUREML_MODEL_DIR is not set)
    if not model_root:
        logger.warning("AZUREML_MODEL_DIR environment variable not set. Falling back to local paths.")
        # Try 'aml_model_files' first, then 'app_artifacts' as per your training script structure
        if os.path.exists(os.path.join("./aml_model_files", "model.pkl")):
            model_root = "./aml_model_files"
            logger.info("Using './aml_model_files' for local model artifacts.")
        elif os.path.exists(os.path.join("./app_artifacts", "model.pkl")):
            model_root = "./app_artifacts" # Fallback if aml_model_files is not found locally for testing
            logger.info("Using './app_artifacts' for local model artifacts.")
        else:
            logger.error("Model artifacts not found in ./aml_model_files or ./app_artifacts for local testing.")
            raise FileNotFoundError("Model artifacts not found for local testing.")
    
    model_path = os.path.join(model_root, "model.pkl")
    feature_order_path = os.path.join(model_root, "model_binary_cols.json")
    disease_names_path = os.path.join(model_root, "model_disease_names.json")

    try:
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")

        logger.info(f"Loading feature order from: {feature_order_path}")
        with open(feature_order_path, 'r') as f:
            feature_order = json.load(f)
        logger.info(f"Feature order loaded successfully. Expected features: {len(feature_order)}")

        logger.info(f"Loading disease names from: {disease_names_path}")
        with open(disease_names_path, 'r') as f:
            disease_names = json.load(f)
        logger.info(f"Disease names loaded successfully. Known diseases: {len(disease_names)}")

    except FileNotFoundError as fnfe:
        logger.error(f"File not found during initialization: {str(fnfe)}. Model root: {model_root}")
        raise
    except Exception as e:
        error_message = f"Error during model/artifact loading: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_message)
        raise RuntimeError(error_message)

    logger.info("Initialization complete.")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    Expected input: A JSON string or dictionary with symptom names as keys and 0 or 1 as values.
    Example: {"itching": 1, "skin_rash": 0, "cough": 1, ...}
    """
    global model, feature_order, disease_names, logger
    logger.info("In run() method. Received new scoring request.")

    try:
        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
                logger.info(f"Successfully parsed raw_data string into JSON. Type: {type(data)}")
            except json.JSONDecodeError as jde:
                logger.error(f"JSONDecodeError: Invalid JSON string received: {raw_data}. Error: {str(jde)}")
                return {"error": f"Invalid JSON format: {str(jde)}", "received_data": raw_data}
        elif isinstance(raw_data, dict):
            data = raw_data
            logger.info(f"Received pre-parsed dictionary. Type: {type(data)}")
        else:
            error_msg = f"Invalid input type. Expected JSON string or dict, got {type(raw_data)}."
            logger.error(error_msg)
            return {"error": error_msg}

        logger.debug(f"Input data for prediction (first 5 items): {dict(list(data.items())[:5])}")

        # Prepare the input DataFrame for the model
        # The model expects features in a specific order (`feature_order`)
        # and all features must be present.
        # Symptoms not provided in the input `data` will default to 0.
        input_vector = [data.get(symptom, 0) for symptom in feature_order]
        
        # Log a snippet of the input vector for debugging, not the whole thing if it's large
        logger.debug(f"Constructed input vector (first 10 features): {input_vector[:10]}")

        input_df = pd.DataFrame([input_vector], columns=feature_order)
        logger.debug(f"Created DataFrame for prediction. Shape: {input_df.shape}")

        # Perform prediction (get probabilities)
        probabilities_array = model.predict_proba(input_df)[0] # Get the first (and only) row
        logger.info("Prediction successful.")

        # Map probabilities to disease names
        # Ensure probabilities are Python floats for JSON serialization
        disease_probs = {disease_names[i]: float(probabilities_array[i]) for i in range(len(disease_names))}
        
        logger.debug(f"Predicted probabilities (first 3): {dict(list(disease_probs.items())[:3])}")
        logger.info("Successfully processed scoring request.")
        
        return disease_probs # AML will serialize this dict to JSON

    except Exception as e:
        error_message = f"Error during scoring: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_message)
        # It's good to return a JSON response for errors too
        return {"error": str(e), "traceback": traceback.format_exc()}

# Example for local testing (optional)
if __name__ == "__main__":
    # This block is for local testing only and won't run in Azure ML
    print("--- Local Test for score.py ---")
    
    # Configure a simple logger for local testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("local_test") # Use a distinct logger for local tests

    # 1. Ensure aml_model_files (or app_artifacts) directory is in the same location as score.py
    #    or adjust the fallback paths in init().
    #    Run train_model.py first to generate these artifacts.
    print("Attempting to initialize...")
    try:
        init()
        print("Initialization successful.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # 2. Create sample input data
    #    Load feature_order to know what symptoms to include
    if feature_order:
        sample_input_dict = {symptom: 0 for symptom in feature_order}
        # Set a few symptoms to 1 for testing
        if len(feature_order) > 5:
            sample_input_dict[feature_order[0]] = 1 # e.g., 'itching': 1
            sample_input_dict[feature_order[3]] = 1 # e.g., 'continuous_sneezing': 1
            sample_input_dict[feature_order[5]] = 1 # e.g., 'chills': 1
        else: # Handle cases with very few features
            for i in range(min(3, len(feature_order))):
                 sample_input_dict[feature_order[i]] = 1

        sample_input_json_str = json.dumps(sample_input_dict)
        print(f"\nSample input (JSON string): {sample_input_json_str[:200]}...") # Print a snippet

        # 3. Test the run function
        print("\nAttempting to run prediction...")
        results = run(sample_input_json_str)
        
        print("\nPrediction results:")
        if isinstance(results, dict) and "error" not in results:
            # Sort by probability for readability in test output
            sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
            for disease, prob in sorted_results[:10]: # Print top 10
                print(f"- {disease}: {prob:.4f}")
        else:
            print(f"Error in prediction: {results}")
    else:
        print("feature_order not loaded, cannot create sample input for local test.")

    print("\n--- Local Test Complete ---")