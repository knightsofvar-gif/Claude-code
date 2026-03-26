#!/usr/bin/env bash
# Installs a nightly cron job to validate the paper recommender at 2am UTC.
# Run this once on your local machine: bash setup_cron.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$HOME/.paper_recommender_cache/validation.log"
CRON_LINE="0 2 * * * cd \"$SCRIPT_DIR\" && python test_recommender.py --skip-doi >> \"$LOG_FILE\" 2>&1"

# Avoid duplicate entries
if crontab -l 2>/dev/null | grep -qF "test_recommender.py"; then
    echo "Cron job already installed."
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Cron job installed. Runs nightly at 2am UTC."
    echo "Logs: $LOG_FILE"
fi

echo ""
echo "To view the current crontab:  crontab -l"
echo "To remove the cron job:       crontab -e  (delete the test_recommender line)"
echo "To run a manual check now:    cd \"$SCRIPT_DIR\" && python test_recommender.py --skip-doi"
