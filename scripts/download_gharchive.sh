#!/usr/bin/env bash
# Download the GH Archive hour files of a period (UTC) into a folder.
#
#   bash scripts/download_gharchive.sh RAW_DATA_FOLDER [START_DATE] [END_DATE_AND_HOUR_EXCLUSIVE]
#   bash scripts/download_gharchive.sh raw_gharchive 2026-06-01 "2026-11-21 12"      # the period of the submitted model
#
# The hour files are named YYYY-MM-DD-H.json.gz (hour without leading zero), as in https://www.gharchive.org .
# Re-runnable: complete files are skipped.  Why a download cannot silently lose an hour:
#   * a file gets its final name only after its download has finished, so a broken download never looks like a finished file;
#   * transient server answers (HTTP 429 / 5xx) are retried, and every hour that is still missing is tried again in up to 3 rounds;
#   * at the end every hour that is still missing is listed with its reason ("GH Archive answers 404" = no such file, or "download failed");
#     the exit code is 1 only if a download failed (run the script again), not if GH Archive itself has no such file.
# (An earlier version ran wget quietly and ignored its errors: a request that failed was silently missing from the folder.)
set -u
OUT=${1:?usage: download_gharchive.sh RAW_DATA_FOLDER [START_DATE] [END_DATE_AND_HOUR]}
START=${2:-2026-06-01}
END=${3:-$(date -u +"%Y-%m-%d %H")}
mkdir -p "$OUT"
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
t=$(date -u -d "$START 00:00" +%s); e=$(date -u -d "$END:00" +%s)
while [ "$t" -lt "$e" ]; do                                  # the file name of every hour of the period
  echo "$(date -u -d @"$t" +%Y-%m-%d)-$(date -u -d @"$t" +%-H).json.gz"
  t=$((t+3600))
done > "$LIST"
missing() { while read -r f; do [ -s "$OUT/$f" ] || echo "$f"; done < "$LIST"; }

# wget does not retry HTTP 429 / 5xx by itself (the option exists since wget 1.19.1)
RETRY=""; wget --help 2>&1 | grep -q -- "--retry-on-http-error" && RETRY="--retry-on-http-error=429,500,502,503,504"
export OUT RETRY
fetch() {                                                    # one hour file: download to a .part file, give it its name when complete
  f=$1
  if wget -q --tries=5 --waitretry=3 $RETRY -O "$OUT/$f.part" "https://data.gharchive.org/$f"; then mv -f "$OUT/$f.part" "$OUT/$f"; else rm -f "$OUT/$f.part"; fi
}
export -f fetch

for round in 1 2 3; do
  todo=$(missing); [ -z "$todo" ] && break
  [ "$round" -gt 1 ] && { echo "round $round: $(echo "$todo" | wc -l) hour file(s) still missing, trying again"; sleep 5; }
  echo "$todo" | xargs -P 16 -n 1 bash -c 'fetch "$0"'
done

left=$(missing); failed=0
echo "files in $OUT: $(ls "$OUT" | grep -c '\.json\.gz$') (the period has $(wc -l < "$LIST") hours)"
if [ -n "$left" ]; then
  echo "NOT in $OUT: $(echo "$left" | wc -l) hour file(s):"
  for f in $left; do
    code=$(wget --spider -S --tries=2 "https://data.gharchive.org/$f" 2>&1 | awk '/^ *HTTP\//{c=$2} END{print c}')
    if [ "$code" = 404 ]; then echo "  $f   GH Archive answers 404: no such file"
    else echo "  $f   download failed (server answer: ${code:-none}): run this script again"; failed=1; fi
  done
else
  echo "all hour files of the period are present"
fi
exit $failed
