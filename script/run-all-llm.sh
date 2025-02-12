#!/bin/bash

# LLM experiments
for config in config/llm/*.yaml; do
    python downstream-baseline.py --config ${config};
done