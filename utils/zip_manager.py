"""
********************************
*   Created by mohammed-alaa   *
********************************
a simple class adds set of files and folders into single zip file .. used extensively in saving checkpoints,logs or predictions
"""
import datetime
import os
import zipfile


class ZipFile:
    def __init__(self, file_name):
        self.zipf = zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED)

    def get_true_size(self):
        size = sum([zinfo.file_size for zinfo in self.zipf.filelist])
        zip_mb = float(size) / 1024 / 1024  # kB
        return zip_mb

    def get_compressed_size(self):
        size = sum([zinfo.compress_size for zinfo in self.zipf.filelist])
        zip_mb = float(size) / 1024 / 1024  # kB
        return zip_mb

    def print_info(self, verbose=False):
        print("%s,total data size is :%.3f mb,compressed :%.3f mb" % (self.zipf.filename, self.get_true_size(), self.get_compressed_size()))
        print("Files are :")
        for info in self.zipf.infolist():
            print(info.filename)
            if verbose:
                print('  Comment     :', info.comment)
                mod_date = datetime.datetime(*info.date_time)
                print('  Modified    :', mod_date)
                if info.create_system == 0:
                    system = 'Windows'
                elif info.create_system == 3:
                    system = 'Unix'
                else:
                    system = 'UNKNOWN'
                print('  System      :', system)
                print('  ZIP version :', info.create_version)

            print('  Compressed  :', info.compress_size, 'bytes')
            print('  Uncompressed:', info.file_size, 'bytes')
            print()

    def add_directory(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                self.zipf.write(os.path.join(root, file))

    def add_file(self, path):
        self.zipf.write(path)

    def __del__(self):
        # self.print_info()
        self.zipf.close()

# import tarfile

# USAGE
# myzipfile = ZipFile("comp.zip")
# myzipfile.addDir('./Bot/')
# #myzipfile.addFile('./Bot/')
# myzipfile.print_info()
#
# for root, dirs, files in os.walk('./Bot'):
#     print((root, dirs, files))
#
#

#
# tar = tarfile.open("TarName.tar.gz", "w:gz")
# tar.add("comp.zip", arcname="comp.zip")
# tar.close()
