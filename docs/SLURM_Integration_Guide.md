# SLURM Integration Guide for PSA Simulations

This guide explains how to run Parameter Sensitivity Analysis (PSA) simulations on SLURM clusters with the new cluster integration functionality.

## Overview

The SLURM integration allows you to scale PSA simulations from local execution to cluster computing while maintaining all existing functionality. Key benefits:

- **Seamless integration**: Add just 2 parameters to existing PSA scripts
- **Automatic job management**: Handles job submission, monitoring, and result collection
- **Fault tolerance**: Automatic retry of failed jobs and partial result recovery
- **Preserved caching**: Maintains existing intelligent caching system
- **Resource efficiency**: Configurable job sizing and resource allocation

## Quick Start

### 1. Basic Local Cluster Usage

```matlab
% Configure SLURM settings
slurm_config = PSA_slurm_config();
slurm_config.partition = 'normal';           % Your cluster partition
slurm_config.patients_per_job = 50;          % Patients per SLURM job

% Run PSA on cluster (add these 2 parameters to existing script)
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'execution_mode', 'slurm', ...           % Enable SLURM
    'slurm_config', slurm_config);           % SLURM config
```

### 2. Remote SSH + Async Usage

```matlab
% Configure for remote cluster access with async execution
slurm_config = PSA_slurm_config();
slurm_config.use_ssh = true;                 % Enable SSH
slurm_config.ssh_host = 'login.cluster.edu'; % Cluster login node
slurm_config.ssh_user = 'username';          % Your username
slurm_config.remote_project_path = '/home/username/QSPIO-PDAC';
slurm_config.async_mode = true;              % Non-blocking execution

% Submit jobs (returns immediately)
[~, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'execution_mode', 'slurm', 'slurm_config', slurm_config);

% Check status anytime
status = PSA_check_async_jobs(metadata.async_state_file);

% Collect results when ready
[results, metadata] = PSA_collect_async_results(metadata.async_state_file);
```

### 3. Complete Examples

- **Local cluster**: `projects/pdac_2025/PSA_script_PDAC_GVAX_slurm_example.m`
- **Remote SSH + Async**: `projects/pdac_2025/PSA_script_PDAC_GVAX_ssh_async_example.m`

## Configuration

### SLURM Configuration Options

```matlab
slurm_config = PSA_slurm_config();

% Basic settings
slurm_config.partition = 'normal';              % SLURM partition
slurm_config.time_limit = '02:00:00';           % Job time limit
slurm_config.memory_per_job = '4G';             % Memory per job  
slurm_config.patients_per_job = 50;             % Patients per job

% Environment
slurm_config.matlab_module = 'matlab/R2023b';  % MATLAB module
slurm_config.matlab_extra_args = '-singleCompThread';

% Resource management
slurm_config.max_concurrent_jobs = 20;         % Max simultaneous jobs
slurm_config.retry_failed_jobs = true;         % Auto-retry failures

% Advanced options
slurm_config.account = 'your_account';         % Billing account
slurm_config.qos = 'normal';                   % Quality of service
slurm_config.sbatch_extra_args = '--exclusive'; % Extra sbatch args

% SSH configuration (for remote clusters)
slurm_config.use_ssh = true;                    % Enable SSH access
slurm_config.ssh_host = 'login.cluster.edu';   % Cluster login node
slurm_config.ssh_user = 'username';             % Your cluster username
slurm_config.ssh_key = '~/.ssh/cluster_key';   % SSH private key path
slurm_config.remote_project_path = '/home/username/QSPIO-PDAC';

% Async execution (non-blocking)
slurm_config.async_mode = true;                 % Return immediately after submission
```

### Choosing Job Size

The key parameter is `patients_per_job`, which determines how to split your cohort:

```matlab
% For 500 patients:
slurm_config.patients_per_job = 50;  % → 10 jobs of 50 patients each
slurm_config.patients_per_job = 100; % → 5 jobs of 100 patients each
slurm_config.patients_per_job = 25;  % → 20 jobs of 25 patients each
```

**Guidelines:**
- **Small jobs (25-50 patients)**: Better fault tolerance, faster queue times
- **Large jobs (100+ patients)**: More efficient resource usage, fewer jobs to manage
- **Consider**: Queue limits, simulation complexity, available memory

## File Organization

The system creates organized directory structures:

```
projects/your_project/
├── slurm_jobs/
│   ├── job_scripts/          # Generated SLURM batch scripts
│   ├── input_chunks/         # Parameter chunks for each job
│   ├── output_chunks/        # Results from individual jobs
│   └── logs/                 # SLURM stdout/stderr logs
└── cache/PSA_simulations/    # Final combined cache (same as local)
```

## Monitoring Jobs

### Built-in Monitoring

**Synchronous jobs:**
```matlab
% Monitor jobs in real-time
PSA_monitor_slurm_jobs(sim_metadata.slurm_jobs);

% Monitor with custom settings
PSA_monitor_slurm_jobs(sim_metadata.slurm_jobs, ...
    'refresh_interval', 15, ...    % Check every 15 seconds
    'show_details', true);         % Show individual job status
```

**Asynchronous jobs:**
```matlab
% Check status anytime (non-blocking)
status = PSA_check_async_jobs(state_file);

% Continuous monitoring loop
while true
    status = PSA_check_async_jobs(state_file, 'verbose', false);
    if status.is_complete, break; end
    pause(300);  % Check every 5 minutes
end

% Collect results when ready
[results, metadata] = PSA_collect_async_results(state_file);
```

### Command Line Monitoring

You can also use standard SLURM commands:

```bash
# Check job status
squeue -u $USER

# Check specific jobs
squeue -j 12345,12346,12347

# View job history
sacct -j 12345 --format=JobID,State,Start,End,Elapsed
```

## Error Handling and Debugging

### Automatic Retry

Failed jobs are automatically retried based on configuration:

```matlab
slurm_config.retry_failed_jobs = true;    % Enable auto-retry
slurm_config.max_retries = 2;             % Maximum attempts
```

### Debugging Failed Jobs

1. **Check SLURM logs**: `projects/your_project/slurm_jobs/logs/`
2. **Review job scripts**: `projects/your_project/slurm_jobs/job_scripts/`
3. **Examine error files**: Look for `error_*.mat` files in output directory

### Common Issues

| Issue | Solution |
|-------|----------|
| "Module not found" | Update `slurm_config.matlab_module` |
| "Permission denied" | Check filesystem permissions or SSH keys |
| "Queue time exceeded" | Increase `slurm_config.time_limit` |
| "Out of memory" | Increase `slurm_config.memory_per_job` |
| "Jobs not starting" | Check partition availability and limits |
| "SSH connection failed" | Verify SSH configuration and keys |
| "Remote path not found" | Check `slurm_config.remote_project_path` |
| "Async jobs lost" | Use the persistent state file to recover |

## Performance Optimization

### Resource Sizing

Monitor resource usage and adjust:

```matlab
% For memory-intensive simulations
slurm_config.memory_per_job = '8G';
slurm_config.patients_per_job = 25;  % Fewer patients per job

% For compute-intensive simulations  
slurm_config.time_limit = '04:00:00';
slurm_config.matlab_extra_args = '-singleCompThread';
```

### Job Sizing Strategy

```matlab
% Conservative approach (good for testing)
slurm_config.patients_per_job = 25;
slurm_config.max_concurrent_jobs = 5;

% Balanced approach (recommended)
slurm_config.patients_per_job = 50;
slurm_config.max_concurrent_jobs = 10;

% Aggressive approach (high-performance clusters)
slurm_config.patients_per_job = 100;
slurm_config.max_concurrent_jobs = 20;
```

## Integration with Existing Workflows

### Minimal Changes Required

Existing PSA scripts need only 2 additional parameters:

```matlab
% OLD: Local execution
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, config);

% NEW: SLURM execution (synchronous)
slurm_config = PSA_slurm_config();
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'execution_mode', 'slurm', 'slurm_config', slurm_config);

% NEW: Remote + Async execution
slurm_config.use_ssh = true;
slurm_config.ssh_host = 'cluster.edu';
slurm_config.async_mode = true;
[~, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'execution_mode', 'slurm', 'slurm_config', slurm_config);
% Later: [results, metadata] = PSA_collect_async_results(metadata.async_state_file);
```

### Backwards Compatibility

- All existing caching functionality preserved
- Same result format and metadata structure
- Existing visualization and analysis code works unchanged
- Can switch between local and SLURM execution seamlessly

### Conditional Execution

```matlab
% Automatic fallback to local execution
if is_slurm_available()
    slurm_config = PSA_slurm_config();
    [results, metadata] = PSA_simulate(..., 'execution_mode', 'slurm', ...
        'slurm_config', slurm_config);
else
    [results, metadata] = PSA_simulate(...);  % Local execution
end
```

## Best Practices

### Before Running

1. **Test locally first**: Verify your PSA script works with a small cohort
2. **Check cluster policies**: Understand queue limits and resource restrictions
3. **Estimate resource needs**: Monitor local execution to size jobs appropriately

### During Execution

1. **Monitor progress**: Use `PSA_monitor_slurm_jobs()` or `squeue`
2. **Check early results**: Verify first few jobs complete successfully
3. **Be patient**: Large cohorts may take hours to complete

### After Completion

1. **Verify results**: Check that expected number of patients completed
2. **Clean up if needed**: Remove temporary files if disk space is limited
3. **Document settings**: Save successful configurations for future use

## Troubleshooting

### Job Submission Fails

```matlab
% Check SLURM availability
[status, output] = system('which sbatch');
if status ~= 0
    error('SLURM not available on this system');
end

% Verify partition exists
[status, output] = system('sinfo -p your_partition');
```

### Jobs Start But Fail Quickly

Check the SLURM error logs in `projects/your_project/slurm_jobs/logs/`. Common issues:
- MATLAB module loading fails
- Path/environment issues
- Memory or time limits exceeded

### Partial Results

The system handles partial completion gracefully:
- Completed jobs are cached and preserved
- Failed jobs can be retried without re-running successful ones
- Use existing caching system to resume interrupted workflows

## Advanced Usage

### Custom Job Scripts

For advanced users, you can modify the generated job scripts by editing the `generate_slurm_script()` function in `PSA_simulate_slurm.m`.

### GPU Support

```matlab
slurm_config.partition = 'gpu';
slurm_config.sbatch_extra_args = '--gres=gpu:1';
```

### High-Memory Jobs

```matlab
slurm_config.partition = 'highmem';
slurm_config.memory_per_job = '32G';
slurm_config.patients_per_job = 200;  % More patients per high-memory job
```

## Support and Feedback

For issues or questions:
1. Check the error logs first
2. Review this documentation  
3. Test with a small cohort to isolate issues
4. Report persistent problems with log files and configuration details