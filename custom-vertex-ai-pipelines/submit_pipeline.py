import warnings
warnings.filterwarnings("ignore", message="Running pip as the 'root' user")

from google.cloud import aiplatform
# Define your project-specific variables
PROJECT_ID = "vertexaipipeline-450203"
REGION = "us-central1"  
BUCKET = "mlworkflow_kubeflowbucket"
PIPELINE_ROOT = f"gs://{BUCKET}/vertex-pipeline-root"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# Create and submit the pipeline job
job = aiplatform.PipelineJob(
    display_name="sample-vertex-ai-pipeline-run",
    template_path="pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "training_data_path": f"gs://{BUCKET}/data/train.csv",
        "model_output_path": f"gs://{BUCKET}/model/model.joblib"
    }
)
job.submit()

job.wait()  # Wait until the job finishes
print("Pipeline run status:", job.state)
