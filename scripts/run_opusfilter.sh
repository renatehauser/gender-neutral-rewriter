#!/bin/bash

################################################
### Filter parallel training data
################################################

set -e

PATH_TO_VIRTUAL_ENVIRONMENT=$1  # e.g. /mnt/work/abmt/venv
SCRIPTS=$2
LANGVAR=$3                      # "neut" or "gen"
STEP=$4                         # "raw" for first filtering step before e.g. RT translation 

source $PATH_TO_VIRTUAL_ENVIRONMENT/bin/activate

opusfilter $SCRIPTS/filter_$LANGVAR.$STEP.yaml