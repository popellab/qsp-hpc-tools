#!/usr/bin/env python3
"""
Interactive setup tool for QSP HPC credentials.

Creates configuration files with proper directory structure:
- Global config: ~/.config/qsp-hpc/credentials.yaml
- Project config: ./batch_credentials.yaml

Usage:
    python setup_credentials.py           # Interactive setup
    python setup_credentials.py --global  # Global config only
    python setup_credentials.py --project # Project config only
"""

import argparse
import sys
from pathlib import Path
import yaml


def get_input(prompt: str, default: str = '', required: bool = False) -> str:
    """Get user input with optional default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        value = input(full_prompt).strip()
        if not value and default:
            return default
        if not value and required:
            print("This field is required. Please enter a value.")
            continue
        return value


def setup_global_config():
    """Set up global HPC credentials."""
    print("\n" + "="*70)
    print("GLOBAL HPC CONFIGURATION")
    print("="*70)
    print("This will create: ~/.config/qsp-hpc/credentials.yaml")
    print("These settings will be used as defaults for all projects.")
    print()

    config = {
        'ssh': {},
        'cluster': {}
    }

    # SSH settings
    print("SSH Configuration:")
    config['ssh']['host'] = get_input("  HPC hostname (e.g., login.hpc.edu)", required=True)
    config['ssh']['user'] = get_input("  SSH username", required=True)
    config['ssh']['key'] = get_input("  SSH key path (leave empty for default)")
    config['ssh']['simulation_pool_path'] = get_input(
        "  Simulation pool path on HPC (e.g., /home/username/qsp-simulations)",
        required=True
    )
    config['ssh']['remote_project_path'] = get_input(
        "  Remote project path (e.g., /home/username/qspio-pdac)"
    )

    # HPC venv path
    config['ssh']['hpc_venv_path'] = get_input(
        "  HPC Python venv path (e.g., /home/username/venv)",
        required=True
    )

    # Cluster settings
    print("\nCluster Configuration:")
    config['cluster']['partition'] = get_input("  SLURM partition", default="shared")
    config['cluster']['time_limit'] = get_input("  Time limit", default="20:00")
    config['cluster']['memory_per_job'] = get_input("  Memory per job", default="2G")
    config['cluster']['matlab_module'] = get_input("  MATLAB module", default="matlab/R2024a")

    # Create config directory
    config_dir = Path.home() / '.config' / 'qsp-hpc'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / 'credentials.yaml'
    
    # Write config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Global configuration saved to: {config_file}")
    return config_file


def setup_project_config():
    """Set up project-specific HPC credentials."""
    print("\n" + "="*70)
    print("PROJECT-SPECIFIC HPC CONFIGURATION")
    print("="*70)
    print("This will create: ./batch_credentials.yaml")
    print("These settings override global defaults for this project only.")
    print()

    # Check if global config exists
    global_config_file = Path.home() / '.config' / 'qsp-hpc' / 'credentials.yaml'
    has_global = global_config_file.exists()
    
    if has_global:
        print("✓ Global config found. Only override settings that differ for this project.")
        print("  (Press Enter to skip a field and use global default)\n")
    else:
        print("⚠ No global config found. You'll need to specify all settings.")
        print("  Consider running with --global first to set defaults.\n")

    config = {
        'ssh': {},
        'cluster': {}
    }

    # SSH settings
    print("SSH Configuration (overrides):")
    host = get_input("  HPC hostname (override)", required=not has_global)
    if host:
        config['ssh']['host'] = host
    
    user = get_input("  SSH username (override)", required=not has_global)
    if user:
        config['ssh']['user'] = user
    
    key = get_input("  SSH key path (override)")
    if key:
        config['ssh']['key'] = key
    
    pool_path = get_input("  Simulation pool path (override)", required=not has_global)
    if pool_path:
        config['ssh']['simulation_pool_path'] = pool_path
    
    project_path = get_input("  Remote project path (override)")
    if project_path:
        config['ssh']['remote_project_path'] = project_path

    venv_path = get_input("  HPC venv path (override)", required=not has_global)
    if venv_path:
        config['ssh']['hpc_venv_path'] = venv_path

    # Cluster settings
    print("\nCluster Configuration (overrides):")
    partition = get_input("  SLURM partition (override)")
    if partition:
        config['cluster']['partition'] = partition
    
    time_limit = get_input("  Time limit (override)")
    if time_limit:
        config['cluster']['time_limit'] = time_limit
    
    memory = get_input("  Memory per job (override)")
    if memory:
        config['cluster']['memory_per_job'] = memory
    
    matlab_module = get_input("  MATLAB module (override)")
    if matlab_module:
        config['cluster']['matlab_module'] = matlab_module

    # Remove empty sections
    if not config['ssh']:
        del config['ssh']
    if not config['cluster']:
        del config['cluster']

    if not config:
        print("\n⚠ No overrides specified. Not creating project config.")
        return None

    config_file = Path('batch_credentials.yaml')
    
    # Write config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Project configuration saved to: {config_file}")
    print("⚠ Remember to add 'batch_credentials.yaml' to .gitignore!")
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description='Interactive setup for QSP HPC credentials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration hierarchy:
  1. ~/.config/qsp-hpc/credentials.yaml (global defaults)
  2. ./batch_credentials.yaml (project overrides)
  3. Constructor argument in code (explicit override)

Examples:
  # Set up global defaults
  python setup_credentials.py --global

  # Set up project-specific overrides
  python setup_credentials.py --project

  # Interactive setup (both)
  python setup_credentials.py
        '''
    )
    parser.add_argument('--global', dest='global_only', action='store_true',
                        help='Set up global config only')
    parser.add_argument('--project', dest='project_only', action='store_true',
                        help='Set up project config only')
    
    args = parser.parse_args()

    print("QSP HPC Credentials Setup")
    print("="*70)

    if args.global_only:
        setup_global_config()
    elif args.project_only:
        setup_project_config()
    else:
        # Interactive - ask what to set up
        print("\nWhat would you like to set up?")
        print("  1. Global config only (~/.config/qsp-hpc/credentials.yaml)")
        print("  2. Project config only (./batch_credentials.yaml)")
        print("  3. Both (recommended for first-time setup)")
        
        choice = get_input("\nChoice [1-3]", default="3", required=True)
        
        if choice == "1":
            setup_global_config()
        elif choice == "2":
            setup_project_config()
        elif choice == "3":
            setup_global_config()
            setup_project_config()
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    print("\n" + "="*70)
    print("Setup complete!")
    print("\nTo use these credentials:")
    print("  from qsp_hpc.batch import HPCJobManager")
    print("  manager = HPCJobManager()  # Automatically loads config")
    print("="*70)


if __name__ == '__main__':
    main()
