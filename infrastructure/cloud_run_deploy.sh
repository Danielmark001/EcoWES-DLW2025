gcloud builds submit --tag gcr.io/$PROJECT_ID/port-energy-management-backend
gcloud run deploy port-energy-management-backend --image gcr.io/$PROJECT_ID/port-energy-management-backend --platform managed

gcloud builds submit --tag gcr.io/$PROJECT_ID/port-energy-management-frontend
gcloud run deploy port-energy-management-frontend --image gcr.io/$PROJECT_ID/port-energy-management-frontend --platform managed
