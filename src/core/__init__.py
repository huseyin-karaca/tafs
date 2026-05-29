"""Paper-agnostic infrastructure for TAFS.

This subpackage hosts the ABCs (models, data, experiments) and utilities
(logging, seeding, metrics, MLflow setup, statistical tests) that TAFS
builds on. TAFS code lives under ``src/tafs/`` and imports from here;
the reverse import direction is not allowed.
"""
