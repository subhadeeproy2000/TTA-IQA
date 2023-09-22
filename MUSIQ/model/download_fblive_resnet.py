import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1sN4IlPnOcbwWvEFNXxFSMV1Oz855FrGL'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
