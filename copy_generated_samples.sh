# copies all generates samples to gcloud bucket
# excludes all files that end in .jpg or .DS_Store
gsutil -m rsync -x ".*\.jpg$|.*\.DS_Store$" -r ./data/output_after_gen/ gs://chris-bachelor-project-bucket/data/output_after_gen/