#!/bin/bash

# LLM experiments
for config in config/llm/*.yaml; do
    python downstream.py --config ${config};
done

# Baseline experiments
for config in config/baselines/*.yaml; do
    python downstream.py --config ${config};
done