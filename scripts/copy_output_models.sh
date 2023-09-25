#!/usr/bin/env bash

# This script copy model results files (without checkpoints) to further processing (table, diagram generation).

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo >&2 "[${date_now}] $1"
}

DIR_IN="out/glue/finetuned_model"
DIR_OUT="out/glue/results-finetuned_model"
OVERWRITE_SAVE_DIR=0

if [ ! -d "${DIR_IN}" ]; then
    log "[ERROR] Input directory: ${DIR_IN} does not exist!"
    exit 2
fi
if [ -d "${DIR_OUT}" ] && [ "${OVERWRITE_SAVE_DIR}" -eq 0 ]; then
    log "[ERROR] Output directory: ${DIR_IN} exists!"
    exit 2
fi

mkdir -p "${DIR_OUT}"

while IFS= read -r -d '' model_dir; do
    dir_name=$(basename "${model_dir}")
    save_dir="${DIR_OUT}/${dir_name}"
    log "[INFO] Reading model results from: ${dir_name}"
    mkdir -p "${save_dir}"
    for file_name_to_copy in 'evaluation_output.json' 'trainer_state.json' 'train_output.json'; do
        cp "${model_dir}/${file_name_to_copy}" "${save_dir}"
    done
done < <(find "${DIR_IN}" -maxdepth 1 -mindepth 1 -type d -print0 | sort --buffer-size=1G --zero-terminated)
