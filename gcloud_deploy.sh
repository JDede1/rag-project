#!/usr/bin/env bash

# ============================================================================
#   FINAL PRODUCTION DEPLOY SCRIPT (CLOUD BUILD + CLOUD RUN)
#   Fully compatible with Google Colab Phase 9 automated deployment
# ============================================================================

set -uo pipefail
# Debug tracing (print every command before executing)
set -x

# ----------------------------------------------------------------------------
# Required environment variables (injected by Colab Phase 9 cell)
# ----------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:?PROJECT_ID not set}"
REGION="${REGION:?REGION not set}"
REPO_NAME="${REPO_NAME:?REPO_NAME not set}"
SERVICE_NAME="${SERVICE_NAME:?SERVICE_NAME not set}"

IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "-----------------------------------------"
echo " PROJECT:        ${PROJECT_ID}"
echo " REGION:         ${REGION}"
echo " REPOSITORY:     ${REPO_NAME}"
echo " SERVICE NAME:   ${SERVICE_NAME}"
echo " DOCKER IMAGE:   ${IMAGE_NAME}"
echo "-----------------------------------------"

# ----------------------------------------------------------------------------
# Enable required GCP services (idempotent)
# ----------------------------------------------------------------------------
gcloud services enable artifactregistry.googleapis.com \
                        run.googleapis.com \
                        cloudbuild.googleapis.com \
                        secretmanager.googleapis.com

# ----------------------------------------------------------------------------
# Create Artifact Registry repo (safe even if exists)
# ----------------------------------------------------------------------------
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="RAG backend Docker images" \
    || echo "Artifact Registry repo already exists."

# ----------------------------------------------------------------------------
# Allow Cloud Run's robot service account to access Secret Manager
# ----------------------------------------------------------------------------
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    || echo "IAM binding already applied."

# ----------------------------------------------------------------------------
# Build and push image using Cloud Build
# ----------------------------------------------------------------------------
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project="${PROJECT_ID}" \
    --substitutions=_AR_IMAGE="${IMAGE_NAME}"

# ----------------------------------------------------------------------------
# DEPLOY to Cloud Run
# We wrap this in a failure-catching block to ensure logs always print
# ----------------------------------------------------------------------------
echo "Starting Cloud Run deployment..."

DEPLOY_OUTPUT=$(mktemp)

if ! gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_NAME}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8000 \
    --set-env-vars=DEPLOY_ENV=cloud \
    --set-env-vars=GEN_MODE=groq \
    --set-secrets=GROQ_API_KEY=GROQ_API_KEY:latest \
    2>&1 | tee "${DEPLOY_OUTPUT}"
then
    echo "---------------------------------------------------"
    echo " Deployment FAILED. Extracting Cloud Run log URL..."
    echo "---------------------------------------------------"

    # Attempt to extract the Cloud Run logs viewer URL
    LOG_URL=$(grep -o "https://console.cloud.google.com/logs/viewer[^ ]*" "${DEPLOY_OUTPUT}" | head -n 1)

    if [[ -n "${LOG_URL}" ]]; then
        echo ""
        echo "Cloud Run Failure Logs:"
        echo "${LOG_URL}"
        echo ""
    else
        echo "Could not automatically extract log URL. Please scroll up for errors."
    fi

    exit 1
fi

# ----------------------------------------------------------------------------
# Success — Fetch deployed Cloud Run URL
# ----------------------------------------------------------------------------
echo "=================================================="
echo " Deployment successful."
echo " Fetching Cloud Run service URL..."
echo "=================================================="

gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format="value(status.url)"

echo "=================================================="
