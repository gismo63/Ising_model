#!/bin/bash
read -p "File name: " name
touch $name
chmod +x $name
gedit $name &
