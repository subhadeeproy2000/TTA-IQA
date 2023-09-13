import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1y4bzTxbECzy2l0Ecu1lqLHEpAMw1thbN'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
