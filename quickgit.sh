#!/bin/bash
read -p "File/Directory name: " name
read -p "Commit message: " mess
git add $name
git commit -m "$mess"
git push origin master
