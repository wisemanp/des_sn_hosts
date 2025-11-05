#!/usr/bin/env python3
import os
import argparse
import logging
from yaml import safe_load as yload
from des_sn_hosts.simulations.scripts.build_sfh_catalog import main as build_sfh_main
from des_sn_hosts.simulations.scripts.build_hosts_from_sfh import main as build_hosts_main
from des_sn_hosts.simulations.scripts.simulate_sne import main as simulate_sne_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Note: importing main() functions and calling them directly isn't ideal for argparse.
# In practice, run these scripts independently, or refactor into callable functions.
# This runner is a placeholder to show orchestration only.


def main():
    ap = argparse.ArgumentParser(description="Run pipeline stages: SFH -> hosts -> SNe")
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--sfh', action='store_true')
    ap.add_argument('--hosts', action='store_true')
    ap.add_argument('--simulate', action='store_true')
    args = ap.parse_args()

    cfg = yload(open(args.config, 'r'))
    out_dir = cfg['save']['dir']
    os.makedirs(out_dir, exist_ok=True)

    logger.info("This runner is a placeholder. Prefer running individual scripts for now:")
    logger.info("  build_sfh_catalog.py -c config.yaml")
    logger.info("  build_hosts_from_sfh.py -c config.yaml --sfh sfh_catalog.h5 [--bake]")
    logger.info("  simulate_sne.py -c aura_config.yaml -n 200000")


if __name__ == '__main__':
    main()
