""" Upload folder to Google Drive

"""
from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from datetime import datetime

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFile, ApiRequestError


def parse_args():
    """ We might want to call from command line and parse arguments as follows

    """
    parser = ArgumentParser(
        description="Upload local folder to Google Drive")
    parser.add_argument('-s', '--source', type=str,
                        help='Folder to upload')
    parser.add_argument('-d', '--destination', type=str,
                        help='Destination Folder in Google Drive')
    parser.add_argument('-p', '--parent', type=str,
                        help='Parent Folder in Google Drive')
    parser.add_argument('-dl', '--just_down', type=bool,
                        help='Whether to just download files')
    parser.add_argument('-ul', '--just_upload', type=bool,
                        help='Whether to just upload files')
    parser.add_argument('-hi', '--hidden', type=bool,
                        help='Whether to sync hidden files')
    parser.add_argument('-c', '--clean_up', type=bool,
                        help='Whether to clean up the remote')
    parser.add_argument('-cp', '--copy', type=bool,
                        help='Whether to create a remote copy')
    parser.add_argument('-sf', '--sync_files', nargs='+',
                        help='Files to sync')

    return parser.parse_args()


def authenticate():
    """ Authenticate to Google API """
    gauth = GoogleAuth()
    return GoogleDrive(gauth)


def get_remote_file_dict(drive, parent_folder_id):

    # Iterate through all files in the parent folder and return title match:
    query = f"'{parent_folder_id}' in parents and trashed=false"
    remote_file_dict = {file['title']: file
                        for file in drive.ListFile({'q': query}).GetList()}
    return remote_file_dict


def get_remote_file(drive, parent_folder_id,
                    file_name, remote_file_dict=None) -> GoogleDriveFile | None:
    """ Check if remote folder exists and return its object (or None if not)

    """
    if remote_file_dict is None:
        remote_file_dict = get_remote_file_dict(drive, parent_folder_id)

    remote_file = remote_file_dict.get(file_name)

    return remote_file


def create_folder(drive, folder_name, parent_folder_id):
    """ Create folder on Google Drive

    """
    folder_metadata = {
        'title': folder_name,
        # Define the file type as folder
        'mimeType': 'application/vnd.google-apps.folder',
        # ID of the parent folder
        'parents': [{"kind": "drive#fileLink", "id": parent_folder_id}]
    }

    folder = drive.CreateFile(folder_metadata)
    folder.Upload()

    return folder


def upload_file(drive, folder_id, file_path, remote_file=None):

    if remote_file is None:
        file = drive.CreateFile({"parents": [{"id": folder_id}],
                                 "title": os.path.split(file_path)[-1]})
    else:
        file = drive.CreateFile({"parents": [{"id": folder_id}],
                                 "title": os.path.split(file_path)[-1],
                                 "id": remote_file['id']})

    file.SetContentFile(file_path)
    file.Upload()


def upload_files(drive, folder_id, local_folder, uploaded_files, ignored_files,
                 sync_files, sync_hidden=False):
    """ Upload files in the local folder to Google Drive

    """
    local_folder_file_list = [os.path.join(local_folder, title)
                              for title in os.listdir(local_folder)]

    remote_file_dict = get_remote_file_dict(drive, folder_id)

    # Auto-iterate through all files in the local folder:

    for local_file_path in local_folder_file_list:

        add_to_last_modified(local_file_path)

        local_file_name = os.path.split(local_file_path)[-1]

        # Continue if file is not in sync_files:
        if sync_files is not None and local_file_path in sync_files:
            continue

        # Continue if file is empty:
        if os.stat(local_file_path) == 0:
            continue

        # Continue if file/folder is ignored:
        if any(x in local_file_path.split("/") for x in ignored_files):
            continue

        # Continue if file/folder is hidden and upload_hidden is false.
        if not sync_hidden and local_file_name[0] == '.':
            continue

        # Get the remote file matching the local file (if not exist -> None):
        remote_file = get_remote_file(drive, folder_id, local_file_name,
                                      remote_file_dict)

        # If file is folder, upload content recursively:
        if os.path.isdir(local_file_path):
            # If the subfolder does not exist, we need to create it:
            if remote_file is None:
                folder = create_folder(drive, local_file_name, folder_id)
                sub_folder_id = folder['id']
            else:
                # Else the remote file is the subfolder, and we get its id:
                sub_folder_id = remote_file['id']
            upload_files(drive, sub_folder_id, local_file_path, uploaded_files,
                         ignored_files, sync_files)
        else:
            # If we found a plain file, the actual file upload happens now:
            last_modified_local = os.path.getmtime(local_file_path)
            if remote_file is None:
                if '.sync' not in local_file_path.split('/'):
                    print('Uploading ' + local_file_path)
                upload_file(drive, folder_id, local_file_path)
                uploaded_files.append(local_file_path)
            else:
                # If remote file exists, check last modification times:
                last_modified_remote = get_last_modified().get(local_file_path)
                # If modification time is unknown, or local is newer, overwrite:
                if (last_modified_remote is None
                        or last_modified_remote < last_modified_local):
                    print('Overwriting remote ' + local_file_path)
                    upload_file(drive, folder_id, local_file_path, remote_file)

    return uploaded_files


def download_files(drive, folder_id, local_folder, uploaded_files,
                   ignored_files, sync_files, sync_hidden=False):
    """ Download files in the local folder from Google Drive

    """
    local_folder_file_list = [os.path.join(local_folder, title)
                              for title in os.listdir(local_folder)]

    remote_file_dict = get_remote_file_dict(drive, folder_id)

    # Auto-iterate through all files in the remote folder:

    for title, file in remote_file_dict.items():

        path = os.path.join(local_folder, title)

        # Continue if file is not in sync_files:
        if sync_files is not None and path in sync_files:
            continue

        # Continue if file/folder is ignored:
        if title in ignored_files:
            continue

        # Continue if file/folder is hidden and upload_hidden is false:
        if not sync_hidden and title[0] == '.':
            continue

        # Continue if files were just uploaded:
        if path in uploaded_files:
            continue

        # If file is folder, upload content recursively:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # If the subfolder does not exist, we need to create it:
            if path not in local_folder_file_list:
                os.mkdir(path)
            download_files(drive, file['id'], path, uploaded_files,
                           ignored_files, sync_files)
        else:
            # If we found a plain file, the actual file upload happens now:
            last_modified_remote = get_last_modified().get(path)
            if path not in local_folder_file_list:
                print('Downloading ' + path)
                file.GetContentFile(path)
                add_to_last_modified(path)
            else:
                # If modification time is unknown, or remote newer, overwrite:
                if (last_modified_remote is None
                        or last_modified_remote > os.path.getmtime(path)):
                    print('Overwriting local ' + path)
                    file.GetContentFile(path)
                    add_to_last_modified(path)


def add_to_last_modified(local_file_path):

    new_entry = {local_file_path: os.path.getmtime(local_file_path)}

    last_modified = get_last_modified()
    last_modified.update(new_entry)

    with open('.last_modified.json', 'w') as file:
        file.write(json.dumps(last_modified))


def get_last_modified():

    last_modified_file_path = ".last_modified.json"

    if os.stat(last_modified_file_path).st_size == 0:
        return {}
    else:
        with open(last_modified_file_path, 'r') as j:
            last_modified = json.loads(j.read())
        return last_modified


def sync_folder(local_folder='.', remote_folder=None, remote_parent='root',
                just_upload=False, just_download=False, sync_hidden=False,
                clean_up_remote=False, create_copy=False, sync_files=None):
    """ Upload an entire folder with all contents to google Drive """

    args = parse_args()

    # Use command line arguments of given, else function call attributes:
    if args.source is not None:
        local_folder = args.source
        if args.destination is None:
            remote_folder = local_folder
        else:
            remote_folder = args.destination
        if args.parent is not None:
            remote_parent = args.parent
        just_upload = False if args.just_upload is None else args.just_upload
        just_download = False if args.just_down is None else args.just_down
        sync_hidden = False if args.hidden is None else args.hidden
        clean_up_remote = False if args.clean_up is None else args.clean_up
        create_copy = False if args.copy is None else args.copy
        sync_files = None if args.sync_files is None else args.sync_files

    if remote_folder is None:
        remote_folder = local_folder

    if local_folder == '.':
        remote_folder = os.getcwd().split('/')[-1]

    if local_folder == '..':
        remote_folder = os.getcwd().split('/')[-2]

    # Authenticate to Google API
    drive = authenticate()

    # Get remote parent folder ID
    if remote_parent == 'root':
        parent_folder_id = 'root'
    else:
        parent_folder = get_remote_file(drive, 'root', remote_parent)
        if parent_folder is None:
            print('Remote parent does not exist, quit program...')
            return 1
        else:
            parent_folder_id = parent_folder['id']

    # Get remote folder ID or create if it does not exist:
    if create_copy:
        today = datetime.today().strftime('%Y-%m-%d')
        remote_folder += ('_' + today)
        folder = get_remote_file(drive, parent_folder_id, remote_folder)
        while folder is not None:
            remote_folder += '-I'
            folder = get_remote_file(drive, parent_folder_id, remote_folder)
    else:
        folder = get_remote_file(drive, parent_folder_id, remote_folder)

    if folder is None:
        print('Creating remote folder ' + remote_folder)
        folder = create_folder(drive, remote_folder, parent_folder_id)
        folder_id = folder['id']
    else:
        folder_id = folder['id']
        if clean_up_remote:
            folder.Delete()

    mod_json = ".last_modified.json"
    # Check if the .sync folder exists in the main local folder (should be
    # since we call this function from within it, but who knows...)
    # Get the .last_modified.json file from it if possible, else we create it:
    sync_config_folder = get_remote_file(drive, folder_id, '.sync')
    if sync_config_folder is not None:
        last_modified_file = get_remote_file(drive, sync_config_folder['id'],
                                             mod_json)
    else:
        last_modified_file = None

    if last_modified_file is not None:
        last_modified_file.GetContentFile(mod_json)

    # We might add files to ignore in .gdriveignore:
    ignored_files = [x.strip() for x in open('.gdriveignore', 'r').readlines()]

    # Finally upload the local files which are not ignored but modified:
    if not just_download:
        try:
            uploaded_files = upload_files(drive, folder_id, local_folder, [],
                                          ignored_files, sync_files,
                                          sync_hidden=sync_hidden)
        except ApiRequestError:
            return -1
    else:
        uploaded_files = []

    # And download the remote ones which are not ignored but modified:
    if not just_upload:
        download_files(drive, folder_id, local_folder, uploaded_files,
                       ignored_files, sync_files, sync_hidden=sync_hidden)

    # Make sure to finally update the .last_modified.json file:
    if sync_config_folder is not None:
        upload_file(drive, sync_config_folder['id'], mod_json,
                    remote_file=last_modified_file)

    return 0


if __name__ == "__main__":
    response = sync_folder()
    # Sometimes we need to run twice, since first run yields error. As a
    # workaround we capture that error, return -1 and restart the process:
    if response == -1:
        print('--- restart required -> issue should be fixed in future ---')
        response = sync_folder()
