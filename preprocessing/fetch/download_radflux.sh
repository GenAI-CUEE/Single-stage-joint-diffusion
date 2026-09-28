#!/bin/bash
# Downloads SIRTA's raw 1-min irradiance files (radflux) via FTP.
#
# Usage: download_radflux.sh <year> <output_dir> [month] [day]
#   <year>        e.g. 2023
#   <output_dir>  local root to mirror into (was hardcoded to ~/SKY/SIRTA/radflux)
#   [month]       e.g. 04 -- mirror only this month (default: all 12 months)
#   [day]         e.g. 01 -- with [month], mirror only this day's folder
#
# Requires SIRTA_FTP_IRRAD_USER in the environment -- SIRTA provides FTP
# access to this archive on request; there is no public/anonymous login, so
# this script refuses to run without it rather than embedding a login.
set -euo pipefail

if [ -z "${SIRTA_FTP_IRRAD_USER:-}" ]; then
	echo "SIRTA_FTP_IRRAD_USER is not set. SIRTA provides FTP access to the" \
	     "radflux irradiance archive on request -- set this environment" \
	     "variable to the login they give you, then re-run this script." >&2
	exit 1
fi

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
	echo "Usage: $0 <year> <output_dir> [month] [day]" >&2
	exit 1
fi

year="$1"
output_dir="$2"
month_arg="${3:-}"
day_arg="${4:-}"

if [ -n "$day_arg" ] && [ -z "$month_arg" ]; then
	echo "Usage: $0 <year> <output_dir> [month] [day] -- [day] requires [month]" >&2
	exit 1
fi

if [ -n "$month_arg" ]; then
	month_list="$month_arg"
else
	month_list="01 02 03 04 05 06 07 08 09 10 11 12"
fi

for month in $month_list;
do
	if [ -n "$day_arg" ]; then
		mkdir -p "$output_dir/$year/$month/$day_arg"
		lftp -c "set ftp:list-options -a;
		open ftp://${SIRTA_FTP_IRRAD_USER}:@ftp.sirta.ipsl.polytechnique.fr;
		lcd $output_dir/$year/$month/$day_arg;
		cd /basesirta/1b/radflux/$year/$month/$day_arg;
		mirror --verbose --exclude-glob .svn" </dev/null
	else
		mkdir -p "$output_dir/$year/$month"
		lftp -c "set ftp:list-options -a;
		open ftp://${SIRTA_FTP_IRRAD_USER}:@ftp.sirta.ipsl.polytechnique.fr;
		lcd $output_dir/$year/$month;
		cd /basesirta/1b/radflux/$year/$month;
		mirror --verbose --exclude-glob .svn" </dev/null
	fi

done
