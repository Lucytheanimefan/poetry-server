from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.CommandLineAuth() #LocalWebserverAuth()

drive = GoogleDrive(gauth)