import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1wzDfk4AkSt-J6lI4r1AXRw8Jmhq6mZTj'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
