#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# Required environment variables (injected from Phase 8 Colab cell)
# -------------------------------------------------------------------
PROJECT_ID=${PROJECT_ID:?PROJECT_ID not set}
REGION=${REGION:?REGION not set}
REPO_NAME=${REPO_NAME:?REPO_NAME not set}
SERVICE_NAME=${SERVICE_NAME:?SERVICE_NAME not set}

IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "-----------------------------------------"
echo " PROJECT:        ${PROJECT_ID}"
echo " REGION:         ${REGION}"
echo " REPOSITORY:     ${REPO_NAME}"
echo " SERVICE NAME:   ${SERVICE_NAME}"
echo " DOCKER IMAGE:   ${IMAGE_NAME}"
echo "-----------------------------------------"

# -------------------------------------------------------------------
# Enable necessary GCP services (idempotent)
# -------------------------------------------------------------------
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com

# -------------------------------------------------------------------
# Create Artifact Registry repo if not exists (safe)
# -------------------------------------------------------------------
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="RAG backend images" \
  || echo "Artifact Registry repo already exists."

# -------------------------------------------------------------------
# Allow Cloud Run service account to access secrets
# -------------------------------------------------------------------
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  || echo "Secret Manager policy already bound."

# -------------------------------------------------------------------
# Build + push image using Cloud Build
# -------------------------------------------------------------------
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project="${PROJECT_ID}" \
  --substitutions=_AR_IMAGE="${IMAGE_NAME}"

# -------------------------------------------------------------------
# Deploy to Cloud Run (CPU, ONNX mode)
# -------------------------------------------------------------------
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --set-env-vars=DEPLOY_ENV=cloud \
  --set-env-vars=GEN_MODE=groq \
  --set-secrets=GROQ_API_KEY=GROQ_API_KEY:latest

echo "========================================="
echo " Deployment complete! Cloud Run URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --format="value(status.url)"
echo "========================================="
