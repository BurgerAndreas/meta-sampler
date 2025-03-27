#!/bin/sh

# for dir in logs/wandb/offline-run*; do
for dir in wandb/offline-run*; do
    if [ -d "$dir" ] && ! ls "$dir"/*.wandb.synced 1> /dev/null 2>&1; then
        wandb sync "$dir"
    fi
done