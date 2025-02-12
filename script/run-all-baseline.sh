#!/bin/bash

# Baseline experiments
for config in config/baselines/*.yaml; do
    python downstream-baseline.py --config ${config};
done