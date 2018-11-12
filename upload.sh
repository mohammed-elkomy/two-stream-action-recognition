#!/usr/bin/env bash
#cp -a  "/media/mohammed-alaa/Core/current tasks/Storage/drive/." ./utils


zip upload.zip -r utils/*.txt *.py */*.py

curl --upload-file ./upload.zip https://transfer.sh/upload.zip --silent

rm upload.zip

#rm ./utils/credentials*.txt
printf "\n"