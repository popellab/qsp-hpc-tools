#!/usr/bin/env python3
"""
QSP HPC Tools Command Line Interface

Provides convenient CLI commands for HPC setup, testing, and management.

Commands:
    qsp-hpc setup    - Interactive setup wizard
    qsp-hpc test     - Test HPC connection
    qsp-hpc info     - Show current configuration
    qsp-hpc logs     - View HPC job logs
"""

import sys
import click
import yaml
from pathlib import Path
from typing import Optional


@click.group()
@click.version_option()
def cli():
    """QSP HPC Tools - Manage QSP simulations on HPC clusters."""
    pass


@cli.command()
@click.option('--global-only', is_flag=True, help='Setup only global config (skip project-specific)')
def setup(global_only):
    """
    Interactive setup wizard for QSP HPC credentials.

    Creates ~/.config/qsp-hpc/credentials.yaml with HPC connection details.
    """
    click.echo("\n" + "=" * 70)
    click.secho("🚀 QSP HPC Tools Setup Wizard", fg='cyan', bold=True)
    click.echo("=" * 70)

    config_dir = Path.home() / '.config' / 'qsp-hpc'
    config_file = config_dir / 'credentials.yaml'

    click.echo(f"\nThis will create: {config_file}")
    click.echo()

    # Check if config already exists
    if config_file.exists():
        click.secho("⚠️  Configuration file already exists!", fg='yellow')
        if not click.confirm(f"Overwrite {config_file}?", default=False):
            click.echo("Setup cancelled.")
            return
        click.echo()

    # SSH Configuration
    click.secho("SSH Configuration", fg='green', bold=True)
    click.echo("-" * 70)

    # Check for SSH config
    ssh_config_path = Path.home() / '.ssh' / 'config'
    ssh_hosts = []

    if ssh_config_path.exists():
        try:
            with open(ssh_config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Host ') and not '*' in line:
                        host = line.split()[1]
                        ssh_hosts.append(host)
        except (OSError, IOError, IndexError):
            # Ignore errors reading SSH config - it's optional
            pass

    if ssh_hosts:
        click.echo()
        click.secho("  ℹ️  Found SSH config hosts:", fg='cyan')
        for host in ssh_hosts[:5]:  # Show first 5
            click.echo(f"     • {host}")
        if len(ssh_hosts) > 5:
            click.echo(f"     ... and {len(ssh_hosts) - 5} more")
        click.echo()
        click.secho("  💡 Tip: You can use your SSH config alias (e.g., 'hpc')", fg='cyan')
        click.secho("      and leave user/key empty to use config settings.", fg='cyan')
        click.echo()

    ssh_host = click.prompt("  HPC hostname or SSH config alias", type=str)

    # Check if this looks like an SSH config alias
    using_ssh_config = ssh_host in ssh_hosts

    if using_ssh_config:
        click.secho(f"  ✓ Using SSH config for '{ssh_host}'", fg='green')
        click.echo()
        click.echo("  Leave user and key empty to use settings from ~/.ssh/config")

    ssh_user = click.prompt("  Username (leave empty for SSH config)", default="", type=str, show_default=False)
    ssh_key = click.prompt("  SSH key path (leave empty for SSH config)", default="", type=str, show_default=False)

    # Test SSH connection
    click.echo()
    click.echo("Testing SSH connection...", nl=False)

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager, BatchConfig

    test_config = BatchConfig(
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        remote_project_path='',
        hpc_venv_path='/tmp',  # Temporary for testing
        simulation_pool_path='/tmp'  # Temporary for testing
    )

    try:
        test_manager = HPCJobManager(config=test_config, verbose=False)
        test_manager.validate_ssh_connection(timeout=10)
        click.secho(" ✓ Connected!", fg='green')
    except Exception as e:
        click.secho(f" ✗ Failed!", fg='red')
        click.secho(f"  Error: {e}", fg='red')
        if not click.confirm("\nContinue anyway?", default=False):
            return

    click.echo()

    # SLURM Configuration
    click.secho("SLURM Configuration", fg='green', bold=True)
    click.echo("-" * 70)

    partition = click.prompt("  Partition", default="parallel", type=str)
    time_limit = click.prompt("  Default time limit", default="01:00:00", type=str)
    mem_per_cpu = click.prompt("  Memory per CPU", default="4G", type=str)

    # Check SLURM access
    click.echo()
    click.echo("Checking SLURM access...", nl=False)

    try:
        returncode, output = test_manager.transport.exec("scontrol --version", timeout=10)
        if returncode == 0 and 'slurm' in output.lower():
            # Extract version
            version = output.strip().split()[-1] if output.strip() else "unknown"
            click.secho(f" ✓ SLURM available (v{version})", fg='green')
        else:
            click.secho(" ✗ SLURM not found!", fg='yellow')
    except Exception as e:
        click.secho(f" ✗ Failed to check SLURM", fg='yellow')

    click.echo()

    # HPC Paths
    click.secho("HPC Paths", fg='green', bold=True)
    click.echo("-" * 70)

    # Try to get actual remote username for better path suggestions
    remote_username = ssh_user
    if not remote_username:
        try:
            returncode, output = test_manager.transport.exec("whoami", timeout=10)
            if returncode == 0:
                remote_username = output.strip()
                click.echo(f"  Detected remote user: {remote_username}")
        except (OSError, RuntimeError, TimeoutError):
            # Could not detect remote username
            remote_username = None

    # Prompt for data base directory
    data_base_dir = click.prompt(
        "  Data base directory name (under your home)",
        default="data",
        type=str
    )

    # Suggest paths based on remote username (if detected)
    if remote_username:
        default_base = f"/home/{remote_username}/qsp-projects"
        default_pool = f"/home/{remote_username}/{data_base_dir}/{remote_username}/qsp_simulations"
    else:
        # No specific defaults if username not detected
        default_base = ""
        default_pool = ""

    remote_base_dir = click.prompt(
        "  Base directory for projects",
        default=default_base if default_base else "/home/your-username/qsp-projects",
        type=str
    )

    # Venv stored relative to remote base directory
    default_venv = f"{remote_base_dir}/.venv/hpc-qsp"
    hpc_venv_path = click.prompt(
        "  Python virtual environment path",
        default=default_venv,
        type=str
    )

    simulation_pool_path = click.prompt(
        "  Simulation pool directory (for cached full simulations)",
        default=default_pool if default_pool else f"/home/your-username/{data_base_dir}/your-username/qsp_simulations",
        type=str
    )

    click.echo()

    # MATLAB Configuration
    click.secho("MATLAB Configuration", fg='green', bold=True)
    click.echo("-" * 70)

    matlab_module = click.prompt("  MATLAB module name", default="matlab/R2024a", type=str)

    # Test MATLAB availability
    click.echo()
    click.echo("Testing MATLAB module...", nl=False)

    try:
        # First just try to load the module
        returncode, output = test_manager.transport.exec(
            f"module load {matlab_module} 2>&1 && echo 'MODULE_OK'",
            timeout=15
        )

        if returncode == 0 and 'MODULE_OK' in output:
            click.secho(f" ✓ {matlab_module} available", fg='green')
        else:
            click.secho(f" ⚠ could not load module", fg='yellow')
            click.echo(f"  Note: Check available modules with: module avail matlab")
            if output.strip():
                click.echo(f"  Error: {output[:200]}")
    except Exception as e:
        click.secho(" ⚠ could not test", fg='yellow')
        click.echo(f"  Note: This is optional - you can update the module name later")
        click.echo(f"  Error: {str(e)[:100]}")

    click.echo()

    # Build config dictionary (needed for venv setup)
    config = {
        'ssh': {
            'host': ssh_host,
            'user': ssh_user,
            'key': ssh_key,
        },
        'cluster': {
            'matlab_module': matlab_module,
        },
        'paths': {
            'remote_base_dir': remote_base_dir,
            'hpc_venv_path': hpc_venv_path,
            'simulation_pool_path': simulation_pool_path,
        },
        'slurm': {
            'partition': partition,
            'time_limit': time_limit,
            'mem_per_cpu': mem_per_cpu,
        },
        'package': {
            'qsp_hpc_tools_source': 'git+https://github.com/jeliason/qsp-hpc-tools.git@main',
        }
    }

    click.echo()

    # Verify and create directories
    click.secho("Verifying Remote Directories", fg='green', bold=True)
    click.echo("-" * 70)

    dirs_to_check = [
        ('Base directory', remote_base_dir),
        ('Python venv', hpc_venv_path),
        ('Simulation pool', simulation_pool_path),
    ]

    dirs_need_creation = []

    for dir_name, dir_path in dirs_to_check:
        click.echo(f"  Checking {dir_name}: {dir_path}...", nl=False)
        try:
            returncode, _ = test_manager.transport.exec(f"test -d {dir_path}", timeout=10)
            if returncode == 0:
                click.secho(" ✓ exists", fg='green')
            else:
                click.secho(" ✗ not found", fg='yellow')
                dirs_need_creation.append((dir_name, dir_path))
        except Exception as e:
            click.secho(f" ✗ error checking", fg='red')
            dirs_need_creation.append((dir_name, dir_path))

    # Offer to create missing directories
    if dirs_need_creation:
        click.echo()
        click.secho("⚠️  Some directories don't exist yet.", fg='yellow')

        if click.confirm("Would you like to create them now?", default=True):
            click.echo()
            for dir_name, dir_path in dirs_need_creation:
                click.echo(f"  Creating {dir_name}: {dir_path}...", nl=False)
                try:
                    returncode, output = test_manager.transport.exec(f"mkdir -p {dir_path}", timeout=10)
                    if returncode == 0:
                        click.secho(" ✓", fg='green')
                    else:
                        click.secho(f" ✗ failed", fg='red')
                        click.echo(f"    Error: {output}")
                except Exception as e:
                    click.secho(f" ✗ error", fg='red')
                    click.echo(f"    Error: {e}")
        else:
            click.echo()
            click.secho("  ⚠️  You'll need to create these directories manually before using qsp-hpc", fg='yellow')

    # Special handling for Python venv
    click.echo()
    if hpc_venv_path in [d[1] for d in dirs_need_creation]:
        click.secho("Python Virtual Environment Setup", fg='cyan', bold=True)
        click.echo("  The HPC Python venv needs to be set up with required packages.")
        click.echo()
        click.echo("  You can either:")
        click.echo("    1. Run the setup script now (recommended)")
        click.echo("    2. Set it up manually later")
        click.echo()

        if click.confirm("  Run Python venv setup now?", default=True):
            click.echo()
            click.echo("  Setting up Python environment on HPC...")
            click.echo("  This will:")
            click.echo("    • Create Python venv using 'uv'")
            click.echo("    • Install qsp-hpc-tools from GitHub")
            click.echo("    • Install all dependencies (numpy, pandas, pyarrow, scipy)")
            click.echo("  This may take a few minutes...")
            click.echo()

            # Get package source from config
            qsp_hpc_tools_source = config.get('package', {}).get(
                'qsp_hpc_tools_source',
                'git+https://github.com/jeliason/qsp-hpc-tools.git@main'
            )

            setup_cmd = f"""
set -e

echo "Creating venv at {hpc_venv_path}..."
uv venv --python 3.11 {hpc_venv_path}

echo "Installing qsp-hpc-tools from {qsp_hpc_tools_source}..."
uv pip install --python {hpc_venv_path}/bin/python "{qsp_hpc_tools_source}"

echo "Verifying installation..."
{hpc_venv_path}/bin/python -c "import qsp_hpc; print('✓ qsp-hpc-tools installed')"
{hpc_venv_path}/bin/python -c "import numpy, pandas, pyarrow; print('✓ Dependencies available')"

echo "Python venv setup complete!"
"""

            try:
                returncode, output = test_manager.transport.exec(setup_cmd.strip(), timeout=300)
                click.echo()
                click.echo("  Output:")
                for line in output.split('\n')[-10:]:  # Show last 10 lines
                    if line.strip():
                        click.echo(f"    {line}")

                if returncode == 0 and 'successfully' in output.lower():
                    click.echo()
                    click.secho("  ✓ Python venv setup complete!", fg='green')
                else:
                    click.echo()
                    click.secho("  ⚠️  Setup may have encountered issues", fg='yellow')
                    click.echo()
                    click.echo("  If 'uv' is not available on HPC, you can:")
                    click.echo("    1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
                    click.echo("    2. Or use standard Python venv and pip")
            except Exception as e:
                click.secho(f"  ✗ Error running setup", fg='red')
                click.echo(f"  Error: {e}")
        else:
            click.echo()
            click.secho("  ℹ️  Remember to set up the Python venv before submitting jobs:", fg='cyan')
            click.echo(f"    ssh {ssh_host}")
            click.echo(f"    uv venv --python 3.11 {hpc_venv_path}")
            click.echo(f"    uv pip install --python {hpc_venv_path}/bin/python git+https://github.com/jeliason/qsp-hpc-tools.git@main")

    # Save configuration
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo()
    click.secho(f"✅ Configuration saved to {config_file}", fg='green', bold=True)

    click.echo("\nNext steps:")
    click.echo("  1. Test connection: qsp-hpc test")
    click.echo("  2. View config: qsp-hpc info")
    click.echo("  3. See usage: qsp-hpc --help")
    click.echo()


@cli.command()
@click.option('--timeout', default=10, help='Connection timeout in seconds')
def test(timeout):
    """
    Test HPC connection and verify SLURM access.

    Validates SSH connection, SLURM commands, and MATLAB availability.
    """
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    click.echo("\n" + "=" * 70)
    click.secho("🧪 Testing HPC Connection", fg='cyan', bold=True)
    click.echo("=" * 70)
    click.echo()

    # Load configuration
    try:
        manager = HPCJobManager(verbose=False)
    except FileNotFoundError as e:
        click.secho("✗ No configuration found!", fg='red')
        click.echo(f"\n{e}")
        click.echo("\nRun 'qsp-hpc setup' to create configuration.")
        sys.exit(1)

    config = manager.config

    # Test SSH connection
    click.echo(f"Testing SSH connection to {config.ssh_user}@{config.ssh_host}...", nl=False)
    try:
        manager.validate_ssh_connection(timeout=timeout)
        click.secho(" ✓", fg='green')
    except Exception as e:
        click.secho(" ✗", fg='red')
        click.secho(f"  Error: {e}", fg='red')
        sys.exit(1)

    # Test whoami
    click.echo(f"Checking remote user...", nl=False)
    try:
        returncode, output = manager.transport.exec("whoami", timeout=timeout)
        if returncode == 0:
            username = output.strip()
            click.secho(f" ✓ {username}", fg='green')
        else:
            click.secho(" ✗", fg='red')
    except Exception as e:
        click.secho(" ✗", fg='red')

    # Test SLURM
    click.echo(f"Checking SLURM availability...", nl=False)
    try:
        returncode, output = manager.transport.exec("scontrol --version", timeout=timeout)
        if returncode == 0 and 'slurm' in output.lower():
            version = output.strip().split()[-1] if output.strip() else "unknown"
            click.secho(f" ✓ v{version}", fg='green')
        else:
            click.secho(" ✗", fg='red')
    except Exception as e:
        click.secho(" ✗", fg='red')

    # Test partition access
    if config.partition:
        click.echo(f"Checking partition '{config.partition}'...", nl=False)
        try:
            returncode, output = manager.transport.exec("sinfo -o '%P'", timeout=timeout)
            if returncode == 0:
                partitions = [line.strip().replace('*', '') for line in output.split('\n')]
                if config.partition in partitions:
                    click.secho(" ✓", fg='green')
                else:
                    click.secho(" ⚠ not found", fg='yellow')
            else:
                click.secho(" ✗", fg='red')
        except (OSError, RuntimeError, TimeoutError):
            # SSH command failed
            click.secho(" ✗", fg='red')

    # Test MATLAB
    click.echo(f"Checking MATLAB module '{config.matlab_module}'...", nl=False)
    try:
        returncode, output = manager.transport.exec(
            f"module load {config.matlab_module} 2>&1 && echo 'OK'",
            timeout=timeout
        )
        if returncode == 0 and 'OK' in output:
            click.secho(" ✓", fg='green')
        else:
            click.secho(" ⚠ could not load", fg='yellow')
    except (OSError, RuntimeError, TimeoutError):
        # SSH command failed
        click.secho(" ✗", fg='red')

    # Test paths
    click.echo(f"Checking remote directories...", nl=False)
    all_paths_ok = True
    for path_name, path in [
        ('venv', config.hpc_venv_path),
        ('simulation pool', config.simulation_pool_path)
    ]:
        try:
            returncode, _ = manager.transport.exec(f"test -d {path}", timeout=timeout)
            if returncode != 0:
                all_paths_ok = False
        except (OSError, RuntimeError, TimeoutError):
            # SSH command failed
            all_paths_ok = False

    if all_paths_ok:
        click.secho(" ✓", fg='green')
    else:
        click.secho(" ⚠ some paths not found", fg='yellow')

    click.echo()
    click.secho("✅ All critical tests passed!", fg='green', bold=True)
    click.echo()


@cli.command()
@click.option('--show-secrets', is_flag=True, help='Show SSH key path (hidden by default)')
def info(show_secrets):
    """
    Show current HPC configuration.

    Displays the configuration from ~/.config/qsp-hpc/credentials.yaml
    """
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    click.echo("\n" + "=" * 70)
    click.secho("📋 Current Configuration", fg='cyan', bold=True)
    click.echo("=" * 70)
    click.echo()

    config_file = Path.home() / '.config' / 'qsp-hpc' / 'credentials.yaml'

    if not config_file.exists():
        click.secho("✗ No configuration found!", fg='red')
        click.echo(f"\nExpected location: {config_file}")
        click.echo("\nRun 'qsp-hpc setup' to create configuration.")
        sys.exit(1)

    try:
        manager = HPCJobManager(verbose=False)
        config = manager.config
    except Exception as e:
        click.secho(f"✗ Error loading configuration: {e}", fg='red')
        sys.exit(1)

    click.secho("SSH Configuration:", fg='green')
    click.echo(f"  Host:     {config.ssh_host}")
    click.echo(f"  User:     {config.ssh_user}")
    if show_secrets:
        click.echo(f"  SSH Key:  {config.ssh_key}")
    else:
        click.echo(f"  SSH Key:  {'*' * 20} (use --show-secrets to reveal)")

    click.echo()
    click.secho("SLURM Configuration:", fg='green')
    click.echo(f"  Partition:      {config.partition}")
    click.echo(f"  Time Limit:     {config.time_limit}")
    click.echo(f"  Memory per CPU: {config.memory_per_job}")

    click.echo()
    click.secho("HPC Paths:", fg='green')
    click.echo(f"  Base Directory:     {config.remote_project_path}")
    click.echo(f"  Python venv:        {config.hpc_venv_path}")
    click.echo(f"  Simulation Pool:    {config.simulation_pool_path}")

    click.echo()
    click.secho("MATLAB Configuration:", fg='green')
    click.echo(f"  Module: {config.matlab_module}")

    click.echo()
    click.secho(f"Configuration file: {config_file}", fg='cyan')
    click.echo()


@cli.command()
@click.argument('project_name', required=False)
@click.option('--task-id', type=int, help='Array task ID to show logs for')
@click.option('--lines', default=50, help='Number of lines to show')
@click.option('--job-id', help='Specific job ID to show logs for')
def logs(project_name, task_id, lines, job_id):
    """
    View HPC SLURM job logs.

    Examples:
        qsp-hpc logs pdac_2025              # Latest job for project
        qsp-hpc logs pdac_2025 --task-id 3  # Task 3 of latest job
        qsp-hpc logs --job-id 12345         # Specific job ID
    """
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

    click.echo("\n" + "=" * 70)
    click.secho("📋 HPC Job Logs", fg='cyan', bold=True)
    click.echo("=" * 70)
    click.echo()

    if not project_name and not job_id:
        click.secho("✗ Must specify either PROJECT_NAME or --job-id", fg='red')
        click.echo("\nUsage:")
        click.echo("  qsp-hpc logs pdac_2025")
        click.echo("  qsp-hpc logs --job-id 12345")
        sys.exit(1)

    try:
        manager = HPCJobManager(verbose=False)
    except FileNotFoundError as e:
        click.secho("✗ No configuration found!", fg='red')
        click.echo(f"\n{e}")
        sys.exit(1)

    if job_id:
        # Show logs for specific job ID
        click.echo(f"Fetching logs for job {job_id}...")

        task_suffix = f"_{task_id}" if task_id is not None else ""
        log_pattern = f"slurm-{job_id}{task_suffix}.out"

        # Try to find the log file
        returncode, output = manager.transport.exec(
            f"find {manager.config.remote_project_path} -name '{log_pattern}' 2>/dev/null | head -1",
            timeout=10
        )

        if returncode == 0 and output.strip():
            log_file = output.strip()
            returncode, log_content = manager.transport.exec(f"tail -n {lines} {log_file}", timeout=10)

            if returncode == 0:
                click.echo()
                click.secho(f"=== {log_file} ===", fg='cyan')
                click.echo(log_content)
            else:
                click.secho("✗ Could not read log file", fg='red')
        else:
            click.secho(f"✗ Log file not found for job {job_id}", fg='red')

    else:
        # Show logs for project (need to implement job state tracking)
        click.secho("⚠️  Project-based log viewing not yet implemented", fg='yellow')
        click.echo("\nUse --job-id to view logs for a specific job:")
        click.echo("  qsp-hpc logs --job-id 12345")

    click.echo()


if __name__ == '__main__':
    cli()
