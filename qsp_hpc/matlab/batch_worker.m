function batch_worker()
%BATCH_WORKER Execute PSA simulation for a chunk of patients (SLURM array job)
%
% This function runs on SLURM compute nodes to process a subset of patients.
% It reads job configuration from JSON and extracts its assigned chunk based on
% SLURM_ARRAY_TASK_ID.
%
% The function expects to find:
%   - Job configuration in 'batch_jobs/input/job_config.json'
%   - Parameter samples CSV in the location specified by job_config.json
%   - Test statistics CSV in the location specified by job_config.json
%
% Outputs:
%   - Saves results to 'batch_jobs/output/chunk_XXX_results.mat'
%   - Saves test statistics to 'batch_jobs/output/chunk_XXX_test_stats.csv'
%   - Saves status to 'batch_jobs/output/chunk_XXX_status.csv'

try
    fprintf('🚀 PSA Worker starting (SLURM array job)\n');
    fprintf('   Node: %s\n', getenv('SLURMD_NODENAME'));
    fprintf('   Job ID: %s\n', getenv('SLURM_JOB_ID'));
    fprintf('   Array Task ID: %s\n', getenv('SLURM_ARRAY_TASK_ID'));

    % Get array index
    array_idx = str2double(getenv('SLURM_ARRAY_TASK_ID'));
    if isnan(array_idx)
        error('Could not read SLURM_ARRAY_TASK_ID environment variable');
    end

    % Set up MATLAB environment
    startup; % Load project paths and settings

    % Determine file paths using absolute paths
    current_dir = pwd;
    base_dir = fullfile(current_dir, 'batch_jobs');
    input_dir = fullfile(base_dir, 'input');
    output_dir = fullfile(base_dir, 'output');

    fprintf('   Working directory: %s\n', current_dir);
    fprintf('   Base directory: %s\n', base_dir);

    % Load job configuration from JSON (replaces PSA setup script and model config)
    job_config_file = fullfile(input_dir, 'job_config.json');
    if ~exist(job_config_file, 'file')
        error('Job config file not found: %s', job_config_file);
    end

    fprintf('   Loading job configuration from JSON...\n');
    fid = fopen(job_config_file, 'r');
    raw_json = fread(fid, inf);
    fclose(fid);
    str_json = char(raw_json');
    job_config = jsondecode(str_json);

    % Extract configuration values
    n_patients = job_config.n_simulations;
    seed = job_config.seed;
    jobs_per_chunk = job_config.jobs_per_chunk;
    model_script = job_config.model_script;
    param_csv_file = fullfile(current_dir, job_config.param_csv);
    test_stats_csv_file = fullfile(current_dir, job_config.test_stats_csv);

    fprintf('   Job config: %d patients, seed=%d, chunk size=%d\n', ...
        n_patients, seed, jobs_per_chunk);
    fprintf('   Model script: %s\n', model_script);

    % Build model_data structure (for compatibility with existing code)
    model_data = struct();
    model_data.config = struct();
    model_data.config.model_script = model_script;

    % Copy dose_schedule and sim_config from job_config if they exist
    if isfield(job_config, 'dose_schedule')
        model_data.dose_schedule = job_config.dose_schedule;
    end
    if isfield(job_config, 'sim_config')
        model_data.sim_config = job_config.sim_config;
    end

    % Recreate model on remote worker instead of using serialized model
    fprintf('   Recreating model on remote worker...\n');

    % Run the model setup script
    eval(model_data.config.model_script);  % Creates 'model' variable

    % Use dose schedule from model_data (if provided), otherwise create default
    if isfield(model_data, 'dose_schedule') && ~isempty(model_data.dose_schedule)
        dose_schedule = model_data.dose_schedule;
        fprintf('   Using uploaded dose schedule:\n');
        % Log dose schedule details
        for i = 1:length(dose_schedule)
            fprintf('     Dose %d: Amount=%.2e, Time=%.1f, Rate=%.2e\n', ...
                i, dose_schedule(i).Amount, dose_schedule(i).Time, dose_schedule(i).Rate);
        end
    else
        % Fallback to baseline (no treatment)
        dose_schedule = [];
        fprintf('   Using baseline dose schedule (no treatment)\n');
    end

    % Apply simulation configuration from model_data or use defaults
    if isfield(model_data, 'sim_config') && ~isempty(model_data.sim_config)
        sim_config = model_data.sim_config;
        fprintf('   Using passed simulation config:\n');
        fprintf('     Solver: %s\n', sim_config.solver);
        fprintf('     Time: %.0f-%.0f %s (daily intervals)\n', ...
            sim_config.start_time, sim_config.stop_time, sim_config.time_units);
        fprintf('     Tolerances: abs=%.2e, rel=%.2e\n', ...
            sim_config.abs_tolerance, sim_config.rel_tolerance);

        % Create time vector - assuming daily intervals
        time_vector = sim_config.start_time:1:sim_config.stop_time;

        model = simulation_config(model, ...
            'solver', sim_config.solver, ...
            'time', time_vector, ...
            'abs_tolerance', sim_config.abs_tolerance, ...
            'rel_tolerance', sim_config.rel_tolerance);
    else
        % Fallback to default configuration
        fprintf('   Using default simulation config (no config passed)\n');
        model = simulation_config(model, ...
            'solver', 'sundials', ...
            'time', 0:1:30, ...
            'abs_tolerance', 1e-12, ...
            'rel_tolerance', 1e-9);
    end

    % Update model_data with fresh objects
    model_data.model = model;
    model_data.dose_schedule = dose_schedule;

    fprintf('   Model recreated successfully\n');

    % Load parameter samples from CSV (always from Python)
    fprintf('   Loading parameters from CSV: %s\n', param_csv_file);
    params_in = load_parameter_samples_csv(param_csv_file);

    fprintf('   Using pre-loaded parameter samples (%d samples)\n', size(params_in.all, 1));

    % Populate LHS fields from params.all matrix (needed for variant creation)
    for i = 1:length(params_in.names)
        pname = params_in.names{i};
        params_in.(pname).LHS = params_in.all(:, i);
    end
    fprintf('   Populated LHS fields for %d parameters\n', length(params_in.names))

    % Calculate my patient range based on array index
    start_patient = array_idx * jobs_per_chunk + 1;
    end_patient = min((array_idx + 1) * jobs_per_chunk, n_patients);
    patient_range = start_patient:end_patient;

    fprintf('   Processing patients %d-%d (%d patients)\n', ...
        start_patient, end_patient, length(patient_range));
    fprintf('   Varying %d parameters\n', length(params_in.names));

    % Extract chunk parameters (keep only my patients)
    chunk_params = struct();
    chunk_params.names = params_in.names;
    for i = 1:length(params_in.names)
        param_name = params_in.names{i};
        if isfield(params_in, param_name) && isfield(params_in.(param_name), 'LHS')
            chunk_params.(param_name) = params_in.(param_name);
            % Subset to my patients
            chunk_params.(param_name).LHS = params_in.(param_name).LHS(patient_range);
        end
    end

    % Run simulations for this chunk
    t_start = tic;
    [chunk_results, chunk_metadata] = run_chunk_simulations(model_data, chunk_params, patient_range);
    t_elapsed = toc(t_start);

    % Save full simulations to HPC persistent storage (if configured)
    if isfield(job_config, 'save_full_simulations') && job_config.save_full_simulations
        fprintf('   Saving full simulations to HPC persistent storage...\n');
        try
            % Extract all species arrays from simulation results
            species_data = extract_all_species_arrays(chunk_results, model);

            % Log species names being saved to Parquet (for debugging)
            fprintf('   Parquet columns (%d species): ', species_data.n_species);
            if species_data.n_species <= 5
                % Show all if 5 or fewer
                fprintf('%s\n', strjoin(species_data.species_names, ', '));
            else
                % Show first 5 and last 2
                first_names = strjoin(species_data.species_names(1:5), ', ');
                last_names = strjoin(species_data.species_names(end-1:end), ', ');
                fprintf('%s ... %s\n', first_names, last_names);
            end

            % Build path to HPC persistent storage from environment variable
            simulation_pool_path = getenv('SIMULATION_POOL_PATH');
            if isempty(simulation_pool_path)
                error('SIMULATION_POOL_PATH environment variable not set. This should be exported by SLURM script.');
            end

            persistent_dir = fullfile(simulation_pool_path, job_config.simulation_pool_id);

            % Create directory if it doesn't exist
            if ~exist(persistent_dir, 'dir')
                mkdir(persistent_dir);
            end

            % Save to Parquet format
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            output_filename = sprintf('batch_%03d_%s_%dsims_seed%d.parquet', ...
                                     array_idx, timestamp, length(patient_range), seed);
            output_file = fullfile(persistent_dir, output_filename);

            save_species_to_parquet(species_data, output_file, chunk_params);

            fprintf('   ✓ Full simulations saved to: %s\n', output_file);

        catch save_err
            fprintf('   ⚠️  Warning: Failed to save full simulations: %s\n', save_err.message);
            % Don't fail the job if full sim saving fails
        end
    end

    % Generic postprocessing based on config
    fprintf('   Postprocessing results...\n');
    outputs = struct();

    % Build postproc_config from JSON configuration
    postproc_config = struct();

    % NOTE: Test statistics extraction is now handled by Python derivation worker
    % (qsp_hpc.batch.derive_test_stats_worker) which processes the saved Parquet files.
    % The MATLAB postproc_extract_test_statistics function is deprecated because:
    % 1. Test statistic functions in CSV are Python code, not MATLAB
    % 2. Python derivation worker runs after all MATLAB jobs complete
    % 3. This provides better error handling and performance
    %
    % The Parquet files saved above contain all species time series needed for
    % Python-based test statistic computation.

    % Save all outputs using generic function
    if ~isempty(fieldnames(outputs))
        postproc_save_outputs(output_dir, array_idx, outputs);
    end

    % Save status vector
    status_csv = sprintf('chunk_%03d_status.csv', array_idx);
    status_csv_file = fullfile(output_dir, status_csv);
    writematrix(chunk_metadata.status, status_csv_file);

    % Save full results file (keep on HPC for later access)
    results_filename = sprintf('chunk_%03d_results.mat', array_idx);
    results_file = fullfile(output_dir, results_filename);
    save(results_file, 'chunk_results', 'chunk_metadata');

    % Summary
    n_success = sum(chunk_metadata.status == 1);
    n_failed_ic = sum(chunk_metadata.status == 0);
    n_failed_sim = sum(chunk_metadata.status == -1);

    fprintf('✅ Chunk processing complete in %.1f seconds (%.2f sec/patient)\n', ...
        t_elapsed, t_elapsed/length(patient_range));
    fprintf('   Success: %d/%d (%.1f%%)\n', n_success, length(patient_range), 100*n_success/length(patient_range));
    fprintf('   Failed IC: %d/%d (%.1f%%)\n', n_failed_ic, length(patient_range), 100*n_failed_ic/length(patient_range));
    fprintf('   Failed sim: %d/%d (%.1f%%)\n', n_failed_sim, length(patient_range), 100*n_failed_sim/length(patient_range));
    fprintf('   Full results: %s\n', results_filename);
    if ~isempty(fieldnames(outputs))
        fprintf('   Outputs: %s\n', strjoin(fieldnames(outputs), ', '));
    end

catch ME
    fprintf('❌ Worker failed with error: %s\n', ME.message);
    fprintf('   Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('     %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end

    % Save error information for debugging (if output_dir exists)
    if exist('output_dir', 'var')
        try
            error_file = fullfile(output_dir, sprintf('error_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
            save(error_file, 'ME');
            fprintf('   Error details saved to: %s\n', error_file);
        catch
            fprintf('   (Could not save error file)\n');
        end
    end

    rethrow(ME);
end

end

function [chunk_results, chunk_metadata] = run_chunk_simulations(model_data, chunk_params, patient_range)
%RUN_CHUNK_SIMULATIONS Execute simulations for parameter chunk
%
% This function mirrors the core simulation loop from PSA_simulate.m
% but operates on a subset of patients.

% Extract model data
model = model_data.model;
dose_schedule = model_data.dose_schedule;
config = model_data.config;

n_patients = length(patient_range);
chunk_results = [];
chunk_metadata = struct();
chunk_metadata.status = zeros(n_patients, 1);  % 1=success, 0=failed_IC, -1=failed_sim
chunk_metadata.patient_range = patient_range;

fprintf('   Starting simulations...\n');
t_sim_start = tic;

for i = 1:n_patients
    patient_global_id = patient_range(i);
    t_patient_start = tic;

    % Progress indicator
    fprintf('     [%s] Patient %d/%d (global ID: %d)\n', ...
        datestr(now, 'HH:MM:SS'), i, n_patients, patient_global_id);

    % Log first 3 parameter values for this patient
    fprintf('       Parameters (first 3): ');
    for j = 1:min(3, length(chunk_params.names))
        pname = chunk_params.names{j};
        if isfield(chunk_params, pname) && isfield(chunk_params.(pname), 'LHS')
            pval = chunk_params.(pname).LHS(i);
            fprintf('%s=%.3e ', pname, pval);
        end
    end
    fprintf('\n       ');

    % Clean up previous model copy
    if i > 1, delete(model_copy); end

    try
        % Create model copy and parameter variant
        model_copy = copyobj(model);
        variant = create_parameter_variant_worker(model_copy, chunk_params, i);

        % Set initial conditions
        try
            [model_copy, ic_success, ~] = initial_conditions(model_copy, 'Variant', variant);

            if ~ic_success
                fprintf('IC failed (no steady state)\n');
                chunk_metadata.status(i) = 0;  % Failed IC
                chunk_results(i).simData = [];
                continue;
            end

        catch ic_err
            fprintf('IC error: %s\n', ic_err.message);
            chunk_metadata.status(i) = 0;  % Failed IC
            chunk_results(i).simData = [];
            continue;
        end

        % Run simulation
        try
            sim_data = sbiosimulate(model_copy, [], variant, dose_schedule);
            t_patient = toc(t_patient_start);
            chunk_metadata.status(i) = 1;   % Success
            chunk_results(i).simData = sim_data;

            fprintf('Success (%.2f sec)\n', t_patient);

        catch sim_err
            fprintf('Simulation error: %s\n', sim_err.message);
            chunk_metadata.status(i) = -1;  % Failed simulation
            chunk_results(i).simData = [];
        end

    catch ME
        fprintf('Unexpected error: %s', ME.message);
        if ~isempty(ME.stack)
            fprintf(' at %s:%d', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\n');
        chunk_metadata.status(i) = -1;  % Failed simulation
        chunk_results(i).simData = [];
    end
end

t_sim_elapsed = toc(t_sim_start);
fprintf('   Simulation loop complete in %.1f seconds\n', t_sim_elapsed);

% Clean up final model copy
if exist('model_copy', 'var')
    delete(model_copy);
end

end

function variant = create_parameter_variant_worker(model, chunk_params, patient_index)
%CREATE_PARAMETER_VARIANT_WORKER Create parameter variant for worker simulation
%
% This function mirrors create_parameter_variant from PSA_simulate.m
% but works with chunk parameters instead of full parameter set.

variant_name = sprintf('patient_%05d', patient_index);
variant = addvariant(model, variant_name);

for j = 1:length(chunk_params.names)
    param_name = chunk_params.names{j};

    % Check if parameter exists in chunk
    if isfield(chunk_params, param_name) && isfield(chunk_params.(param_name), 'LHS')
        param_value = chunk_params.(param_name).LHS(patient_index);

        % Check if parameter exists in model
        param_obj = sbioselect(model, 'Type', 'parameter', 'Name', param_name);
        compartment_obj = sbioselect(model, 'Type', 'compartment', 'Name', param_name);

        if ~isempty(param_obj)
            addcontent(variant, {'parameter', param_name, 'Value', param_value});
        elseif ~isempty(compartment_obj)
            addcontent(variant, {'compartment', param_name, 'Capacity', param_value});
        end
    end
end

end
