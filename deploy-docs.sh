#!/bin/bash
set -e
LOG=/var/log/deploy-docs.log
echo "[$(date -Is)] Pulling..." >> "$LOG"
cd /var/www/trydal.io/docs-source && git pull origin main >> "$LOG" 2>&1
cd /var/www/trydal.io/phd-workspace && git pull origin main >> "$LOG" 2>&1
ln -sf /var/www/trydal.io/phd-workspace /var/www/trydal.io/docs-source/docs/phd-workspace
cd /var/www/trydal.io/docs-source && /root/.local/bin/uv run mkdocs build >> "$LOG" 2>&1
echo "[$(date -Is)] Done." >> "$LOG"
