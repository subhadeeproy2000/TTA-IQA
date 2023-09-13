import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1-IY3VJDsUYZJ2LDTzDcUf_ujDY7NAmSt'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
