import gdown

# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1iPIMl1k51bNjNcaXXmqdwNs0iSi42kpX'

# Download the file from Google Drive
gdown.download(file_url, quiet=False)
