#!/bin/bash
fileid="1XcfCdr3B2GUMlo69h0eN-x-OkpjU_j4L"
filename="concatenated_abridged.jsonl"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
