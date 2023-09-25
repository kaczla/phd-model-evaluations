#!/usr/bin/env bash

set -e

DATASET_PATH='glue-lm-gap'
SOURCE_SPLIT_NAME='dev-0'
TARGET_SPLIT_NAME='dev-1k'
NUMBER_SAMPLES='1000'

if [ ! -d "${DATASET_PATH}" ]; then
    echo >&2 "[ERROR] Cannot find dataset directory: ${DATASET_PATH}"
    exit 1
fi

SOURCE_SPLIT_PATH="${DATASET_PATH}/${SOURCE_SPLIT_NAME}"
if [ ! -d "${DATASET_PATH}" ]; then
    echo >&2 "[ERROR] Cannot find set: ${SOURCE_SPLIT_NAME} in ${SOURCE_SPLIT_PATH}"
    exit 1
fi

TARGET_SPLIT_PATH="${DATASET_PATH}/${TARGET_SPLIT_NAME}"
mkdir -p "${TARGET_SPLIT_PATH}"

paste <(xzcat "${SOURCE_SPLIT_PATH}/in.tsv.xz") <(xzcat "${SOURCE_SPLIT_PATH}/expected.tsv.xz") <(xzcat "${SOURCE_SPLIT_PATH}/raw_data.txt.xz") | shuf -n "${NUMBER_SAMPLES}" >"${TARGET_SPLIT_PATH}/all_data.tsv"

cut -f 1-3 "${TARGET_SPLIT_PATH}/all_data.tsv" | xz >"${TARGET_SPLIT_PATH}/in.tsv.xz"
cut -f 4 "${TARGET_SPLIT_PATH}/all_data.tsv" | xz >"${TARGET_SPLIT_PATH}/expected.tsv.xz"
cut -f 5 "${TARGET_SPLIT_PATH}/all_data.tsv" | xz >"${TARGET_SPLIT_PATH}/raw_data.tsv.xz"

rm "${TARGET_SPLIT_PATH}/all_data.tsv"
