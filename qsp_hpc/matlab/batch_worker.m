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
    t_startup = tic;
    startup; % Load project paths and settings
    dt_startup = toc(t_startup);
    fprintf('   [timing] startup: %.2fs\n', dt_startup);

    % Open parpool if MATLAB_WORKERS > 0 (enables parfor over patients)
    matlab_workers = str2double(getenv('MATLAB_WORKERS'));
    if isnan(matlab_workers) || matlab_workers < 0
        matlab_workers = 0;
    end
    if matlab_workers > 0
        p_existing = gcp('nocreate');
        if isempty(p_existing) || p_existing.NumWorkers ~= matlab_workers
            if ~isempty(p_existing)
                delete(p_existing);
            end
            fprintf('   Opening parpool with %d workers...\n', matlab_workers);
            t_pool = tic;
            parpool('local', matlab_workers);
            fprintf('   ✓ parpool ready in %.1fs\n', toc(t_pool));
        end
    end

    % Determine file paths
    current_dir = pwd;
    base_dir = fullfile(current_dir, 'batch_jobs');
    input_dir = getenv('BATCH_INPUT_DIR');
    if isempty(input_dir)
        input_dir = fullfile(base_dir, 'input');
    end
    output_dir = fullfile(base_dir, 'output');

    fprintf('   Working directory: %s\n', current_dir);
    fprintf('   Input directory: %s\n', input_dir);

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
    if startsWith(job_config.param_csv, '/')
        param_csv_file = job_config.param_csv;
    else
        param_csv_file = fullfile(current_dir, job_config.param_csv);
    end

    fprintf('   Job config: %d patients, seed=%d, chunk size=%d\n', ...
        n_patients, seed, jobs_per_chunk);
    fprintf('   Model script: %s\n', model_script);

    % Build model_data structure (for compatibility with existing code)
    model_data = struct();
    model_data.config = struct();
    model_data.config.model_script = model_script;

    % Copy dosing and sim_config from job_config if they exist
    if isfield(job_config, 'dosing')
        model_data.dosing = job_config.dosing;
    end
    if isfield(job_config, 'sim_config')
        model_data.sim_config = job_config.sim_config;
    end

    % Recreate model on remote worker instead of using serialized model
    fprintf('   Recreating model on remote worker...\n');

    % Run the model setup script
    t_model_build = tic;
    eval(model_data.config.model_script);  % Creates 'model' variable
    dt_model_build = toc(t_model_build);
    fprintf('   [timing] model build (eval script): %.2fs\n', dt_model_build);

    % Create dose schedule from dosing config (if provided)
    if isfield(model_data, 'dosing') && ~isempty(model_data.dosing)
        dosing = model_data.dosing;
        if isfield(dosing, 'drugs') && ~isempty(dosing.drugs)
            fprintf('   Creating dose schedule for: %s\n', strjoin(dosing.drugs, ', '));

            % Build name-value pairs for schedule_dosing()
            dosing_args = {};
            fields = fieldnames(dosing);
            for i = 1:length(fields)
                field = fields{i};
                if ~strcmp(field, 'drugs')
                    % Convert field name to schedule_dosing format (e.g., GVAX_dose)
                    dosing_args{end+1} = field;
                    dosing_args{end+1} = dosing.(field);
                end
            end

            % Call schedule_dosing with drug names and parameters
            dose_schedule = schedule_dosing(dosing.drugs, dosing_args{:});

            % Log dose schedule details
            fprintf('   Dose schedule created:\n');
            for i = 1:length(dose_schedule)
                fprintf('     %s: Amount=%.2e, StartTime=%.1f, Interval=%.1f, Repeat=%d\n', ...
                    dose_schedule(i).Name, dose_schedule(i).Amount, ...
                    dose_schedule(i).StartTime, dose_schedule(i).Interval, ...
                    dose_schedule(i).RepeatCount);
            end
        else
            % No drugs specified - baseline scenario
            dose_schedule = [];
            fprintf('   Using baseline dose schedule (no drugs)\n');
        end
    else
        % Fallback to baseline (no treatment)
        dose_schedule = [];
        fprintf('   Using baseline dose schedule (no dosing config)\n');
    end

    % Apply simulation configuration from model_data or use defaults
    t_sim_config = tic;
    if isfield(model_data, 'sim_config') && ~isempty(model_data.sim_config)
        sim_config = model_data.sim_config;
        fprintf('   Using passed simulation config:\n');
        fprintf('     Solver: %s\n', sim_config.solver);
        fprintf('     Time: %.0f-%.0f %s (half-day intervals)\n', ...
            sim_config.start_time, sim_config.stop_time, sim_config.time_units);
        fprintf('     Tolerances: abs=%.2e, rel=%.2e\n', ...
            sim_config.abs_tolerance, sim_config.rel_tolerance);

        % Create time vector with half-day intervals (matches run_median_simulation)
        time_vector = sim_config.start_time:0.5:sim_config.stop_time;

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
    dt_sim_config = toc(t_sim_config);
    fprintf('   [timing] simulation_config: %.2fs\n', dt_sim_config);

    % Apply model acceleration if requested (compile to C for faster simulation)
    dt_accel = 0;
    accel_requested = isfield(model_data, 'sim_config') && isfield(model_data.sim_config, 'accelerate_model') && model_data.sim_config.accelerate_model;
    if accel_requested
        fprintf('   Accelerating model with sbioaccelerate...\n');
        t_accel_start = tic;
        try
            sbioaccelerate(model);
            dt_accel = toc(t_accel_start);
            fprintf('   ✓ Model accelerated in %.1f seconds\n', dt_accel);
        catch accel_err
            dt_accel = toc(t_accel_start);
            fprintf('   ⚠️  sbioaccelerate failed after %.1fs: %s (continuing without acceleration)\n', dt_accel, accel_err.message);
        end
    end
    fprintf('   [timing] sbioaccelerate: %.2fs (requested=%d)\n', dt_accel, accel_requested);

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

    % Subset sample_indices for this chunk so save_species_to_parquet can
    % stamp each row with its global sample_index (used downstream for
    % cross-scenario alignment).
    if isfield(params_in, 'sample_indices') && ~isempty(params_in.sample_indices)
        chunk_params.sample_indices = params_in.sample_indices(patient_range);
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
    % status convention: 0=success, 1=failure (any kind). The IC-vs-sim
    % split lives only in the per-patient log lines now.
    n_success = sum(chunk_metadata.status == 0);
    n_failed = sum(chunk_metadata.status == 1);

    fprintf('✅ Chunk processing complete in %.1f seconds (%.2f sec/patient)\n', ...
        t_elapsed, t_elapsed/length(patient_range));
    fprintf('   [timing-summary] startup=%.2fs model_build=%.2fs sim_config=%.2fs sbioaccelerate=%.2fs parfor_total=%.2fs n_patients=%d\n', ...
        dt_startup, dt_model_build, dt_sim_config, dt_accel, t_elapsed, length(patient_range));
    fprintf('   Success: %d/%d (%.1f%%)\n', n_success, length(patient_range), 100*n_success/length(patient_range));
    fprintf('   Failed:  %d/%d (%.1f%%)  (see per-patient log lines for IC vs sim split)\n', ...
        n_failed, length(patient_range), 100*n_failed/length(patient_range));
    fprintf('   Full results: %s\n', results_filename);
    if ~isempty(fieldnames(outputs))
        fprintf('   Outputs: %s\n', strjoin(fieldnames(outputs), ', '));
    end

    % Explicit parpool shutdown. Without this, MATLAB hangs on `exit` after the
    % worker finishes, holding the compute node until SLURM timeout (observed on
    % ~25% of tasks after parfor_patient_loop + parfor_extraction PRs).
    close_parpool_if_open();

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

    close_parpool_if_open();
    rethrow(ME);
end
end


function close_parpool_if_open()
%CLOSE_PARPOOL_IF_OPEN Best-effort shutdown of the current parpool.
%
% Called at the end of batch_worker so MATLAB can exit cleanly. Without this,
% MATLAB frequently hangs on `exit` waiting for worker processes to drain,
% holding the SLURM allocation until its wall-clock limit.
    try
        p = gcp('nocreate');
        if ~isempty(p)
            fprintf('   Closing parpool (NumWorkers=%d)...\n', p.NumWorkers);
            t_close = tic;
            delete(p);
            fprintf('   ✓ parpool closed in %.1fs\n', toc(t_close));
        end
    catch close_err
        fprintf('   parpool close warning: %s\n', close_err.message);
    end

end

function [chunk_results, chunk_metadata] = run_chunk_simulations(model_data, chunk_params, patient_range)
%RUN_CHUNK_SIMULATIONS Execute simulations for parameter chunk
%
% Uses parfor when MATLAB_WORKERS env var > 0 (a parpool must already be open,
% typically opened in batch_worker). When MATLAB_WORKERS is 0/unset, runs
% serially. Either way, iterations are independent — each creates its own
% `copyobj(model)` and variant.

% Determine parfor worker count from env (0 = serial)
num_workers = str2double(getenv('MATLAB_WORKERS'));
if isnan(num_workers) || num_workers < 0
    num_workers = 0;
end

% Extract and broadcast model data (parfor's static analyzer needs explicit names)
model_bcast = model_data.model;
dose_bcast = model_data.dose_schedule;
params_bcast = chunk_params;
patient_range_bcast = patient_range;
if isfield(model_data, 'sim_config')
    sim_cfg_bcast = model_data.sim_config;
else
    sim_cfg_bcast = struct();
end
if isfield(sim_cfg_bcast, 'initialization_function')
    init_func_name = sim_cfg_bcast.initialization_function;
else
    init_func_name = '';
end

n_patients = length(patient_range);

% Pre-allocate sliced outputs (parfor-safe)
sim_data_cell = cell(n_patients, 1);
% status convention: 0=success, 1=failure (any kind). Collapsed from the
% old tri-state {1,0,-1} encoding so MATLAB and C++ Parquets agree on the
% filter the Python derive_test_stats_worker uses (status==0 => process).
% Specific failure mode is still visible in the per-patient log lines.
status_arr = ones(n_patients, 1);  % default = failure; flipped to 0 on success

fprintf('   Starting simulations (parfor workers=%d, %d patients)...\n', ...
    num_workers, n_patients);
t_sim_start = tic;

parfor (i = 1:n_patients, num_workers)
    t_patient_start = tic;
    patient_global_id = patient_range_bcast(i);

    % Suppress SimBiology complex-number warning during ODE integration on
    % each worker (fires millions of times from TCR quadratic / ^(2/3) terms).
    warning('off', 'SimBiology:SimFunction:COMPLEX_DATA'); %#ok<PFOUS>

    fprintf('     Patient %d/%d (global ID: %d) starting\n', i, n_patients, patient_global_id);

    status_i = 1;  % default = failure; only set to 0 on a clean simulate()
    sim_data_i = [];
    model_copy = [];
    try
        model_copy = copyobj(model_bcast);
        variant = create_parameter_variant_worker(model_copy, params_bcast, i);

        % Initialize model if an init function was configured
        ic_ok = true;
        if ~isempty(init_func_name)
            try
                [model_copy, ic_success, ~] = feval(init_func_name, model_copy, 'Variant', variant);
                if ~ic_success
                    fprintf('     Patient %d: IC rejected (%s, %.2fs)\n', ...
                        i, init_func_name, toc(t_patient_start));
                    ic_ok = false;
                end
            catch ic_err
                fprintf('     Patient %d: IC error (%s): %s\n', i, init_func_name, ic_err.message);
                ic_ok = false;
            end
        end

        if ic_ok
            try
                sim_data_i = sbiosimulate(model_copy, [], variant, dose_bcast);
                status_i = 0;
                fprintf('     Patient %d: ok (%.2fs)\n', i, toc(t_patient_start));
            catch sim_err
                fprintf('     Patient %d: sim error: %s\n', i, sim_err.message);
            end
        end
    catch ME
        fprintf('     Patient %d: unexpected error: %s\n', i, ME.message);
    end

    % Clean up this iteration's model copy
    if ~isempty(model_copy) && isvalid(model_copy)
        delete(model_copy);
    end

    status_arr(i) = status_i;
    sim_data_cell{i} = sim_data_i;
end

% Reassemble chunk_results as struct array (matches pre-parfor interface)
chunk_results = struct([]);
for i = 1:n_patients
    chunk_results(i).simData = sim_data_cell{i};
end

chunk_metadata = struct();
chunk_metadata.status = status_arr;
chunk_metadata.patient_range = patient_range;

t_sim_elapsed = toc(t_sim_start);
fprintf('   Simulation loop complete in %.1f seconds\n', t_sim_elapsed);

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
