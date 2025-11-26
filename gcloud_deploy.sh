#!/usr/bin/env bash
set -euo pipefail

# ============================
# CONFIGURATION (Auto-read env)
# ============================
PROJECT_ID=${PROJECT_ID:-"YOUR_PROJECT_ID"}
REGION=${REGION:-"us-central1"}
REPO_NAME=${REPO_NAME:-"rag-repo"}
SERVICE_NAME=${SERVICE_NAME:-"rag-backend"}

IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "-----------------------------------------"
echo " PROJECT:        ${PROJECT_ID}"
echo " REGION:         ${REGION}"
echo " REPOSITORY:     ${REPO_NAME}"
echo " SERVICE NAME:   ${SERVICE_NAME}"
echo " DOCKER IMAGE:   ${IMAGE_NAME}"
echo "-----------------------------------------"


# ============================
# Enable required GCP services
# ============================
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com


# ============================
# Create repo if missing
# ============================
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="RAG backend images" \
  || echo "Artifact Registry repo already exists."


# ============================
# Verify Groq secret exists
# ============================
echo "Checking GROQ_API_KEY secret..."
if ! gcloud secrets describe GROQ_API_KEY --project "${PROJECT_ID}" > /dev/null 2>&1; then
  echo "ERROR: Secret GROQ_API_KEY does NOT exist!"
  echo "Create it using:"
  echo "  gcloud secrets create GROQ_API_KEY --data-file=your_key.txt"
  exit 1
fi


# ============================
# Grant Cloud Run secret access
# ============================
echo "Granting secret access to Cloud Run..."

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" \
  --format="value(projectNumber)")

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"


# ============================
# Build & Push Docker image
# ============================
echo "Submitting build to Cloud Build..."
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project="${PROJECT_ID}" \
  --substitutions=_AR_IMAGE="${IMAGE_NAME}"


# ============================
# Deploy to Cloud Run
# ============================
echo "Deploying to Cloud Run..."

gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --set-env-vars=GEN_MODE=groq \
  --set-secrets=GROQ_API_KEY=GROQ_API_KEY:latest


# ============================
# Output deployed URL
# ============================
echo "========================================="
echo " Deployment complete!"
echo " Cloud Run URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --format="value(status.url)"
echo "========================================="
