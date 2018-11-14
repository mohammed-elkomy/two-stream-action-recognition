"""
********************************
*   Created by mohammed-alaa   *
********************************
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DISCLAIMER:
this class can access/delete/create your drive account by providing a credentials file >not< your mail and PW of course
You have to revise this class to make sure it doesn't do any illegal things to your data >> it's really safe don't worry :)
********************************
drive manager: connects with drive api on colab which have very high speed network :D 30 MBps :D
You need credentials files from drive api
This can upload and download files from your drive and save check points while training on Colab efficiently and resume your training on any other machine :3

IMPORTANT these files are confidential don't share them or they have control on your drive account !
********************************
To use this:
1.just create a folder on your drive and replace personal_dfolder
you need to know base_download_dfolder is owned my mohammed.a.elkomy and you can access all of its public experiments (the code can access it and you can resume from my experiments :D)

2.obtain your credential file/files and name them as cred* for example cred1.txt cred_mohammed.txt or any thing cred*.txt
3.you can use multiple credentials files = multiple drive accounts which matches cred*
4.it will automatically push your checkpoints regularly to your personal_dfolder and it can if needed download experiments from base_download_dfolder
********************************
"""

import glob
import os
import shutil
import threading
import time
import zipfile

from pydrive.auth import GoogleAuth  # from google
from pydrive.drive import GoogleDrive  # from google

from utils.zip_manager import ZipFile


class DriveManager:
    # check projects directories
    def __init__(self, project_name,
                 personal_dfolder="1sjKuoFGYqURSiCOM-gUQ4iuyx4M0pJxd",  # your folder the code can read and write to it using your own credentials
                 base_download_dfolder="1B82anWV8Mb4iHYmOp9tIR9aOTlfllwsD",  # my folder..the code have read access to it(public)..don't change this <<<
                 cred_dir='./utils'
                 ):
        # ----------------------------------------------------------------------------------------
        # Authenticate and create the PyDrive client.
        # This only needs to be done once per notebook.
        gauth = GoogleAuth()

        self.cred_files = sorted(glob.glob(os.path.join(cred_dir, "cred*")))

        gauth.LoadCredentialsFile(self.cred_files[0])  # we need this
        print("Using {} as the main credentials file".format(self.cred_files[0]))
        self.drive = GoogleDrive(gauth)

        self.base_projects_dfolder = base_download_dfolder  # this contains my experiments
        self.personal_dfolder = personal_dfolder  # make your own projects folder

        self.project_name = project_name
        self.project_id = self.make_sure_project()
        self.cred_dir = cred_dir
        print("Total Available space from my drive", self.available_space())

    def get_projects_list(self, base_folder):
        return self.drive.ListFile({'q': 'trashed=false and "{}" in parents and mimeType = "application/vnd.google-apps.folder"'.format(base_folder)}).GetList()

    def is_project_exists(self, project_name):
        komy_projects_list = self.get_projects_list(self.base_projects_dfolder)
        my_projects_list = self.get_projects_list(self.personal_dfolder)

        return {"owned by komy": len(list(file for file in komy_projects_list if file["title"] == project_name)) > 0,
                "owned by me": len(list(file for file in my_projects_list if file["title"] == project_name)) > 0,
                }

    def make_sure_project(self):
        if not self.is_project_exists(self.project_name)["owned by me"]:
            print("Creating new project:", self.project_name)
            folder_metadata = {'title': self.project_name, 'mimeType': 'application/vnd.google-apps.folder', "parents": [{"kind": "self.drive#fileLink", "id": self.personal_dfolder}]}
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()

        time.sleep(10)  # make sure it's created

        komy_projects_list = self.get_projects_list(self.base_projects_dfolder)
        is_new_to_komy = len(list(file for file in komy_projects_list if file["title"] == self.project_name)) == 0
        my_projects_list = self.get_projects_list(self.personal_dfolder)

        return {"owned by komy": None if is_new_to_komy else list(file for file in komy_projects_list if file["title"] == self.project_name)[0]['id'],
                "owned by me": list(file for file in my_projects_list if file["title"] == self.project_name)[0]['id'],
                }

    def available_space(self):
        return sum(map(lambda item: 15 - item[1], self.used_per_account()))

    def used_per_account(self):
        used_space = []
        for chosen_cred in self.cred_files:
            gauth = GoogleAuth()
            gauth.LoadCredentialsFile(chosen_cred)
            drive = GoogleDrive(gauth)

            total = 0
            for i in drive.ListFile({'q': "mimeType != 'application/vnd.google-apps.folder' and 'me' in owners"}).GetList():
                if "fileSize" in i.keys():
                    total += int(i["fileSize"])

            total /= 1024 ** 3
            used_space.append((chosen_cred, total))

        return sorted(used_space, key=lambda item: item[1])

    def search_file(self, file_name):

        return self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType != 'application/vnd.google-apps.folder' and  '{}' in parents".format(file_name, self.project_id["owned by me"])}).GetList() + \
               self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType != 'application/vnd.google-apps.folder' and  '{}' in parents".format(file_name, self.project_id["owned by komy"])}).GetList()

    def search_folder(self, folder_name):

        return self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType = 'application/vnd.google-apps.folder' and  '{}' in parents".format(folder_name, self.project_id["owned by me"])}).GetList() + \
               self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType = 'application/vnd.google-apps.folder' and  '{}' in parents".format(folder_name, self.project_id["owned by komy"])}).GetList()

    def _upload_file(self, file_path):
        chosen_cred, _ = self.used_per_account()[0]

        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(chosen_cred)
        self.drive = GoogleDrive(gauth)

        upload_started = time.time()
        title = os.path.split(file_path)[-1]
        uploaded = self.drive.CreateFile({'title': title, "parents": [{"kind": "drive#fileLink", "id": self.project_id["owned by me"]}]})
        uploaded.SetContentFile(file_path)  # file on disk
        uploaded.Upload()
        print("cred file", chosen_cred)

        self.log_upload_drive(uploaded.get('id'), title, upload_started)

    def upload_project_files(self, files_list, snapshot_name, dir_list=None):
        """
        Compresses list of files/dirs and upload them asynchronously to drive as .zip
        """
        if dir_list is None:
            dir_list = []
        snapshot_name += ".zip"

        def upload_job():
            """
            uploads single zip file containing checkpoint/logs
            """
            snapshot_zip = ZipFile(snapshot_name)

            for file in files_list:
                snapshot_zip.add_file(file)

            for dir in dir_list:
                snapshot_zip.add_directory(dir)

            snapshot_zip.print_info()

            snapshot_zip.zipf.close()  # now i can upload
            self._upload_file(snapshot_name)

            try:
                for dir in dir_list:
                    shutil.rmtree(dir)

            except:
                pass

        upload_thread = threading.Thread(target=upload_job)
        upload_thread.start()

    def upload_project_file(self, file_path):
        """
        upload a file asynchronously to drive
        """
        upload_thread = threading.Thread(target=lambda: self._upload_file(file_path))
        upload_thread.start()

    def download_file(self, file_id, save_path, unzip=True, replace=True):
        download_started = time.time()
        # gauth = GoogleAuth()
        # gauth.LoadCredentialsFile(self.cred_file_base.format(random.randint(self.start, self.end)))
        # self.drive = GoogleDrive(gauth)

        downloaded = self.drive.CreateFile({'id': file_id})

        if not replace:
            if os.path.isfile(save_path):
                local_files = glob.glob(save_path + ".*")
                if local_files:
                    last = int(sorted(
                        glob.glob(save_path + ".*"), key=lambda path: int(path.split(".")[-1])
                    )[-1].split(".")[-1])
                    save_path += "." + str(last + 1)
                else:
                    save_path += ".1"

        downloaded.GetContentFile(save_path)  # Download file and save locally
        if unzip:
            zip_ref = zipfile.ZipFile(save_path, 'r')
            zip_ref.extractall('.')
            zip_ref.close()
        self.log_download_drive(downloaded['id'], downloaded['title'], save_path, download_started)

        return save_path

    def download_project_files(self, unzip=False, replace=True):
        my_project_files = self.list_project_files_owned_by_me()
        if my_project_files:
            project_files = my_project_files
        else:
            project_files = self.list_project_files_owned_by_komy()

        self.download_files_list(project_files, unzip, replace)

    def download_files_list(self, project_files, unzip, replace):
        id_name_list = []

        for project_file in project_files:
            id_name_list.append((project_file["id"], project_file["title"]))

        for file_id, file_path in id_name_list:
            self.download_file(file_id, file_path, unzip, replace)

    def list_project_files_owned_by_komy(self):
        if self.project_id["owned by komy"] is None:
            return []
        return self.drive.ListFile({'q': "trashed=false and mimeType != 'application/vnd.google-apps.folder' and '{}' in parents".format(self.project_id["owned by komy"])}).GetList()

    def list_project_files_owned_by_me(self):
        return self.drive.ListFile({'q': "trashed=false and mimeType != 'application/vnd.google-apps.folder' and '{}' in parents".format(self.project_id["owned by me"])}).GetList()

    def list_project_files_owned_by_this_cred_file(self, drive):
        return drive.ListFile({'q': "mimeType != 'application/vnd.google-apps.folder' and '{}' in parents and 'me' in owners".format(self.project_id["owned by me"])}).GetList()

    def delete_project_files(self):
        for chosen_cred in self.cred_files:
            gauth = GoogleAuth()
            gauth.LoadCredentialsFile(chosen_cred)
            drive = GoogleDrive(gauth)

            for project_file in self.list_project_files_owned_by_this_cred_file(drive):
                project_file.Delete()
                print(project_file["title"], "deleted")

        project_folder = self.drive.CreateFile({'id': self.project_id["owned by me"]})
        project_folder.Delete()

    def list_projects(self):
        return {"owned by komy": self.drive.ListFile({'q': "trashed=false and mimeType = 'application/vnd.google-apps.folder' and '{}' in parents".format(self.base_projects_dfolder)}).GetList(),
                "owned by me": self.drive.ListFile({'q': "trashed=false and mimeType = 'application/vnd.google-apps.folder' and '{}' in parents".format(self.personal_dfolder)}).GetList(),
                }

    def get_latest_snapshot_meta(self):
        if len(self.list_project_files_owned_by_me()) > 0:
            return True, self.list_project_files_owned_by_me()[0]['title'], self.list_project_files_owned_by_me()[0]['id']
        elif len(self.list_project_files_owned_by_komy()) > 0:
            return True, self.list_project_files_owned_by_komy()[0]['title'], self.list_project_files_owned_by_komy()[0]['id']
        else:
            return False, None, None

    def get_latest_snapshot(self):

        if_possible, save_path, file_id = self.get_latest_snapshot_meta()
        if if_possible:
            save_path = self.download_file(file_id, save_path)
        return if_possible, save_path

    def time_taken(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        return int(hours), int(minutes), int(seconds)

    def log_download_drive(self, _id, title, saved_as, download_start_time):
        hours, minutes, seconds = self.time_taken(download_start_time, time.time())
        print("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        print('Download and unzipped file with ID:{}, titled:{}, saved as:{}'.format(_id, title, saved_as))
        print("*" * 100)

    def log_upload_drive(self, _id, title, upload_start_time):
        hours, minutes, seconds = self.time_taken(upload_start_time, time.time())
        print("Took {:0>2}:{:0>2}:{:0>2} to ".format(hours, minutes, seconds), end="")
        print('Upload file with ID:{}, titled:{}'.format(_id, title))
        print("*" * 100)


if __name__ == '__main__':
    # "mot-xception-adam-5e-05-imnet"
    # "mot-xception-adam-5e-06-imnet"
    # "spa-xception-adam-5e-06-imnet"
    #  spa-xception-adam-5e-05-imnet
    print(sorted(glob.glob("utils/cred*")))

    for project in ["heavy-spa-xception-adam-1e-05-imnet"]:
        drive_manager = DriveManager(project, cred_dir="utils")  # , "credentials{}.txt"
        print(drive_manager.list_projects())
        continue
        # drive_manager.upload_file("150-0.86043-0.85858.zip")
        # drive_manager.upload_file("155-0.86043-0.85937.zip")

        # for i in drive_manager.list_projects():
        #     print(
        #         i['title']
        #     )

        # drive_manager.download_project_files(unzip=False)

        pprint.pprint(drive_manager.used_per_account())
        print(
            drive_manager.available_space()
        )

        # drive_manager.delete_project_files()

        print(
            drive_manager.available_space()
        )
        print(
            drive_manager.list_project_files_owned_by_komy())
        print(
            drive_manager.list_project_files_owned_by_me()
        )
        print(
            drive_manager.list_project_files_owned_by_this_cred_file(drive_manager.drive)
        )
