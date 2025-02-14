from kfp.dsl import component, pipeline

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model_op(training_data_path: str, model_output_path: str):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib
    try:
        # Dummy training for demonstration:
        print("Reading training data from:", training_data_path)
        data = pd.read_csv(training_data_path)
        print("Successfully read the data. Columns:", data.columns)
        
        # Check if the expected column 'target' exists
        if "target" not in data.columns:
            raise ValueError("Expected column 'target' not found in the training data.")
        
        X = data.drop("target", axis=1)
        y = data["target"]
        print("Training model...")
        
        model = LogisticRegression()
        model.fit(X, y)
        print("Model training completed.")

        print("Saving model to:", model_output_path)
        joblib.dump(model, model_output_path)
        print("Training complete. Model saved successfully.")
        
    except Exception as e:
        print("An error occurred in train_model_op:")
        print(e)
        # Optionally, you can log more detailed traceback information
        import traceback
        traceback.print_exc()
        # Re-raise the exception so that the pipeline run fails with a clear error message.
        raise

@pipeline(
    name="sample-vertex-ai-pipeline",
    pipeline_root="gs://mlworkflow_kubeflowbucket/vertex-pipeline-root"
)
def my_pipeline(
    training_data_path: str = "gs://mlworkflow_kubeflowbucket/data/train.csv",
    model_output_path: str = "gs://mlworkflow_kubeflowbucket/model/model.joblib"
):
    train_model_op(
        training_data_path=training_data_path,
        model_output_path=model_output_path
    )

