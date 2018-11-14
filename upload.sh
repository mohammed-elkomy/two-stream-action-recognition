#!/usr/bin/env bash

# get drive credentials files
cp -a  "/media/mohammed-alaa/Core/current tasks/Storage/drive/." ./utils

# create zipped file of the code
zip upload.zip -r utils/*.txt *.py */*.py

# use transfer sh to upload the zipped file
curl --upload-file ./upload.zip https://transfer.sh/upload.zip --silent

# clean and print the link
rm upload.zip

rm ./utils/cred*.txt
printf "\n"