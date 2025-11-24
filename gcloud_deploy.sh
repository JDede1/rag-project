#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
PROJECT_ID=${PROJECT_ID:-"YOUR_PROJECT_ID"}
REGION=${REGION:-"YOUR_REGION"}           # e.g. us-central1
REPO_NAME=${REPO_NAME:-"rag-repo"}
SERVICE_NAME=${SERVICE_NAME:-"rag-backend"}
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
# ------------------------

echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Repo:    ${REPO_NAME}"
echo "Service: ${SERVICE_NAME}"
echo "Image:   ${IMAGE_NAME}"

# Enable APIs
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com

# Create Artifact Registry repo if not exists
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="RAG backend images" || echo "Repo may already exist."

# Submit build to Cloud Build (using cloudbuild.yaml)
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project="${PROJECT_ID}" \
  --substitutions=_AR_IMAGE="${IMAGE_NAME}"

# Deploy to Cloud Run
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --platform=managed \
  --allow-unauthenticated \
  --region="${REGION}" \
  --port=8000
