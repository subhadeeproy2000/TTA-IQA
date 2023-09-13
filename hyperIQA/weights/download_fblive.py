import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1yvLdBVh-kATCAHpfBRQwz3ERTKgGaUAv'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
