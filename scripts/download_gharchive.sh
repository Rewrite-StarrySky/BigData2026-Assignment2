#!/usr/bin/env bash
# Download the GH Archive hour files of a period (UTC) into a folder.  Existing files are skipped, hours that GH Archive does not have (404) are skipped.
#
#   bash scripts/download_gharchive.sh RAW_DATA_FOLDER [START_DATE] [END_DATE_AND_HOUR_EXCLUSIVE]
#   bash scripts/download_gharchive.sh raw_gharchive 2026-06-01 "2026-11-21 14"      # the period of the submitted model
#
# The hour files are named YYYY-MM-DD-H.json.gz (hour without leading zero), as in https://www.gharchive.org .
set -u
OUT=${1:?usage: download_gharchive.sh RAW_DATA_FOLDER [START_DATE] [END_DATE_AND_HOUR]}
START=${2:-2026-06-01}
END=${3:-$(date -u +"%Y-%m-%d %H")}
mkdir -p "$OUT"
t=$(date -u -d "$START 00:00" +%s); e=$(date -u -d "$END:00" +%s)
while [ "$t" -lt "$e" ]; do
  d=$(date -u -d @"$t" +%Y-%m-%d); h=$(date -u -d @"$t" +%-H)
  f="$OUT/$d-$h.json.gz"
  [ -s "$f" ] || echo "https://data.gharchive.org/$d-$h.json.gz"
  t=$((t+3600))
done | xargs -P 16 -n 1 wget -q -nc -P "$OUT" --tries=5 --waitretry=3 || true
echo "files in $OUT: $(ls "$OUT" | wc -l)"
