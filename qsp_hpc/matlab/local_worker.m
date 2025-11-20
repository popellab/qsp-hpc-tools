function local_worker(config_json_path, output_csv_path)
%LOCAL_WORKER Execute QSP simulations locally (no SLURM required)
%
% This function runs QSP simulations on a local machine using MATLAB,
% without requiring HPC infrastructure. It's designed for testing,
% debugging, and small-scale simulations.
%
% Inputs:
%   config_json_path  - Path to JSON configuration file
%   output_csv_path   - Path where test statistics CSV will be written
%
% Configuration JSON format:
%   {
%     "model_script": "model_setup_script_name",
%     "param_csv": "/path/to/parameters.csv",
%     "test_stats_csv": "/path/to/test_statistics.csv",
%     "n_simulations": 10,
%     "seed": 2025,
%     "model_version": "v1",
%     "scenario": "baseline",
%     "dose_schedule": {...},  // optional
%     "sim_config": {...}      // optional
%   }
%
% Output:
%   - Writes test statistics to output_csv_path (CSV format, no header)
%   - Each row corresponds to one simulation
%   - Columns correspond to test statistics in test_stats_csv order

try
    fprintf('🔬 Local QSP Worker Starting\n');
    fprintf('   Config: %s\n', config_json_path);
    fprintf('   Output: %s\n', output_csv_path);

    % Load configuration from JSON
    if ~exist(config_json_path, 'file')
        error('Config file not found: %s', config_json_path);
    end

    fid = fopen(config_json_path, 'r');
    raw_json = fread(fid, inf);
    fclose(fid);
    str_json = char(raw_json');
    config = jsondecode(str_json);

    % Extract configuration
    model_script = config.model_script;
    param_csv = config.param_csv;
    test_stats_csv = config.test_stats_csv;
    n_sims = config.n_simulations;
    seed = config.seed;

    fprintf('   Model: %s\n', model_script);
    fprintf('   Simulations: %d\n', n_sims);
    fprintf('   Seed: %d\n', seed);

    % Set up model
    fprintf('   Setting up model...\n');
    try
        eval(model_script);  % Creates 'model' variable
        fprintf('   ✓ Model loaded: %s\n', model.Name);
    catch ME
        error('Failed to load model script "%s": %s', model_script, ME.message);
    end

    % Configure dose schedule
    if isfield(config, 'dose_schedule') && ~isempty(config.dose_schedule)
        dose_schedule = config.dose_schedule;
        fprintf('   Using custom dose schedule (%d doses)\n', length(dose_schedule));
    else
        dose_schedule = [];
        fprintf('   Using baseline (no treatment)\n');
    end

    % Configure simulation settings
    if isfield(config, 'sim_config') && ~isempty(config.sim_config)
        sim_config = config.sim_config;
        fprintf('   Custom sim config: solver=%s, time=%.0f-%.0f\n', ...
            sim_config.solver, sim_config.start_time, sim_config.stop_time);

        time_vector = sim_config.start_time:1:sim_config.stop_time;
        model = simulation_config(model, ...
            'solver', sim_config.solver, ...
            'time', time_vector, ...
            'abs_tolerance', sim_config.abs_tolerance, ...
            'rel_tolerance', sim_config.rel_tolerance);
    else
        % Default configuration
        fprintf('   Using default sim config\n');
        model = simulation_config(model, ...
            'solver', 'sundials', ...
            'time', 0:1:30, ...
            'abs_tolerance', 1e-12, ...
            'rel_tolerance', 1e-9);
    end

    % Load parameters from CSV
    fprintf('   Loading parameters from CSV...\n');
    if ~exist(param_csv, 'file')
        error('Parameter CSV not found: %s', param_csv);
    end
    params = load_parameter_samples_csv(param_csv);

    if size(params.all, 1) ~= n_sims
        error('Parameter CSV has %d rows but config specifies %d simulations', ...
            size(params.all, 1), n_sims);
    end

    % Populate LHS fields (required for variant creation)
    for i = 1:length(params.names)
        pname = params.names{i};
        params.(pname).LHS = params.all(:, i);
    end

    fprintf('   ✓ Loaded %d parameters for %d simulations\n', ...
        length(params.names), n_sims);

    % Run simulations
    fprintf('   Running simulations...\n');
    t_start = tic;
    results = struct('simData', cell(n_sims, 1));
    status = zeros(n_sims, 1);  % 1=success, 0=failed_IC, -1=failed_sim

    for i = 1:n_sims
        fprintf('     [%d/%d] ', i, n_sims);

        try
            % Create model copy and variant
            model_copy = copyobj(model);
            variant = create_local_variant(model_copy, params, i);

            % Set initial conditions
            try
                [model_copy, ic_success, ~] = initial_conditions(model_copy, 'Variant', variant);

                if ~ic_success
                    fprintf('IC failed\n');
                    status(i) = 0;
                    results(i).simData = [];
                    delete(model_copy);
                    continue;
                end
            catch ic_err
                fprintf('IC error: %s\n', ic_err.message);
                status(i) = 0;
                results(i).simData = [];
                delete(model_copy);
                continue;
            end

            % Run simulation
            try
                sim_data = sbiosimulate(model_copy, [], variant, dose_schedule);
                status(i) = 1;
                results(i).simData = sim_data;
                fprintf('Success\n');
            catch sim_err
                fprintf('Sim error: %s\n', sim_err.message);
                status(i) = -1;
                results(i).simData = [];
            end

            delete(model_copy);

        catch ME
            fprintf('Error: %s\n', ME.message);
            status(i) = -1;
            results(i).simData = [];
        end
    end

    t_elapsed = toc(t_start);
    n_success = sum(status == 1);
    n_failed_ic = sum(status == 0);
    n_failed_sim = sum(status == -1);

    fprintf('   ✓ Simulations complete in %.1f sec (%.2f sec/sim)\n', ...
        t_elapsed, t_elapsed/n_sims);
    fprintf('     Success: %d/%d (%.1f%%)\n', n_success, n_sims, 100*n_success/n_sims);
    fprintf('     Failed IC: %d/%d\n', n_failed_ic, n_sims);
    fprintf('     Failed sim: %d/%d\n', n_failed_sim, n_sims);

    % Extract test statistics
    fprintf('   Extracting test statistics...\n');
    if ~exist(test_stats_csv, 'file')
        error('Test statistics CSV not found: %s', test_stats_csv);
    end

    % Use existing postprocessing function
    postproc_config = struct();
    postproc_config.test_stats = struct();
    postproc_config.test_stats.csv_file = test_stats_csv;

    [test_stats_data, test_stat_ids, ~] = postproc_extract_test_statistics(...
        results, model, postproc_config.test_stats, fileparts(test_stats_csv));

    if isempty(test_stats_data)
        error('Failed to extract test statistics');
    end

    fprintf('   ✓ Extracted %d test statistics for %d simulations\n', ...
        size(test_stats_data, 2), size(test_stats_data, 1));

    % Write output CSV (no header, just data)
    writematrix(test_stats_data, output_csv_path);
    fprintf('   ✓ Wrote test statistics to: %s\n', output_csv_path);

    fprintf('✅ Local worker complete\n');

catch ME
    fprintf('❌ Local worker failed: %s\n', ME.message);
    fprintf('   Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('     %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

end

function variant = create_local_variant(model, params, sim_index)
%CREATE_LOCAL_VARIANT Create parameter variant for local simulation
%
% Args:
%   model: SimBiology model object
%   params: Parameter structure with names and LHS fields
%   sim_index: Index of current simulation (1-based)
%
% Returns:
%   variant: SimBiology variant object with parameters set

variant_name = sprintf('local_sim_%05d', sim_index);
variant = addvariant(model, variant_name);

for j = 1:length(params.names)
    param_name = params.names{j};

    if isfield(params, param_name) && isfield(params.(param_name), 'LHS')
        param_value = params.(param_name).LHS(sim_index);

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
