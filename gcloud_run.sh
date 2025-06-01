gcloud auth login

gcloud config set project tailoredtastetuner

gcloud config get-value project

gcloud builds submit --tag gcr.io/tailoredtastetuner/tailored-taste-tuner

gcloud run deploy tailored-taste-tuner \
  --image gcr.io/tailoredtastetuner/tailored-taste-tuner \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 4 \
  --min-instances 0