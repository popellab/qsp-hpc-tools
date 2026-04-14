function save_species_to_parquet(species_data, output_file, chunk_params)
%SAVE_SPECIES_TO_PARQUET Save extracted species data to Parquet format via Python
%
% This function converts MATLAB species data to JSON, then calls a Python
% script to write the data in Parquet format for efficient storage.
%
% Inputs:
%   species_data  - Struct from extract_all_species_arrays with fields:
%                   .n_sims, .n_species, .species_names, .time_arrays,
%                   .species_arrays, .status
%   output_file   - Path to output Parquet file
%   chunk_params  - (Optional) Struct with parameter names and values:
%                   .names - Cell array of parameter names
%                   .(param_name).LHS - Array of parameter values for each simulation
%
% Example:
%   save_species_to_parquet(species_data, '/path/to/output.parquet', chunk_params);

fprintf('   Saving species data to Parquet: %s\n', output_file);

% Create temporary JSON file for data transfer
temp_json = tempname;
temp_json = [temp_json '.json'];

% Set up cleanup to delete temp file when function exits (success or error)
cleanupObj = onCleanup(@() deleteIfExists(temp_json));

try
    % Convert MATLAB cell arrays to nested JSON structure
    json_data = struct();
    json_data.n_sims = species_data.n_sims;
    json_data.n_species = species_data.n_species;
    json_data.species_names = species_data.species_names;
    json_data.status = species_data.status;

    % Extract and include parameter data if provided
    if nargin >= 3 && ~isempty(chunk_params) && isfield(chunk_params, 'names')
        fprintf('   Including %d parameters in Parquet file\n', length(chunk_params.names));
        json_data.param_names = chunk_params.names;

        % Extract parameter values into matrix (n_sims x n_params)
        param_values = zeros(species_data.n_sims, length(chunk_params.names));
        for j = 1:length(chunk_params.names)
            param_name = chunk_params.names{j};
            if isfield(chunk_params, param_name) && isfield(chunk_params.(param_name), 'LHS')
                param_values(:, j) = chunk_params.(param_name).LHS;
            end
        end
        json_data.param_values = param_values;
    else
        % No parameters provided
        json_data.param_names = {};
        json_data.param_values = [];
    end

    % Stamp each row with its global sample_index from the theta pool so
    % downstream loaders can intersect scenarios by index, not row position.
    if nargin >= 3 && isfield(chunk_params, 'sample_indices') && ~isempty(chunk_params.sample_indices)
        json_data.sample_indices = double(chunk_params.sample_indices(:));
    else
        json_data.sample_indices = [];
    end

    % Convert time arrays (cell array -> array of arrays)
    json_data.time_arrays = cell(species_data.n_sims, 1);
    for i = 1:species_data.n_sims
        if isempty(species_data.time_arrays{i})
            json_data.time_arrays{i} = [];
        else
            json_data.time_arrays{i} = species_data.time_arrays{i};
        end
    end

    % Convert species arrays (n_sims x n_species cell array -> nested arrays)
    json_data.species_arrays = cell(species_data.n_sims, 1);
    for i = 1:species_data.n_sims
        json_data.species_arrays{i} = cell(species_data.n_species, 1);
        for j = 1:species_data.n_species
            if isempty(species_data.species_arrays{i, j})
                json_data.species_arrays{i}{j} = [];
            else
                json_data.species_arrays{i}{j} = species_data.species_arrays{i, j};
            end
        end
    end

    % Write JSON file
    json_str = jsonencode(json_data);
    fid = fopen(temp_json, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);

    % Find Python script relative to this MATLAB file
    % save_species_to_parquet.m is at: qsp_hpc/matlab/
    % write_species_parquet.py is at: qsp_hpc/simulation/
    matlab_dir = fileparts(mfilename('fullpath'));
    python_script = fullfile(matlab_dir, '..', 'simulation', 'write_species_parquet.py');

    if ~exist(python_script, 'file')
        error('Failed to locate write_species_parquet.py at: %s', python_script);
    end

    % Determine which Python to use
    % 1. Try HPC venv from environment variable
    % 2. Fall back to system Python
    hpc_venv_path = getenv('HPC_VENV_PATH');
    if ~isempty(hpc_venv_path)
        venv_python = fullfile(hpc_venv_path, 'bin', 'python');
        if exist(venv_python, 'file')
            python_cmd = venv_python;
            fprintf('   Using HPC venv Python: %s\n', python_cmd);
        else
            python_cmd = 'python';
            fprintf('   Using system Python\n');
        end
    else
        python_cmd = 'python';
        fprintf('   Using system Python\n');
    end

    cmd = sprintf('"%s" "%s" "%s" "%s"', python_cmd, python_script, temp_json, output_file);
    [status, output] = system(cmd);

    if status ~= 0
        error('Python Parquet writer failed:\n%s', output);
    end

    fprintf('   Parquet file saved: %s\n', output_file);

catch ME
    fprintf('   ⚠️  Error saving Parquet: %s\n', ME.message);
    rethrow(ME);
end

end

function deleteIfExists(filepath)
%DELETEIFEXISTS Delete file if it exists (helper for onCleanup)
    if exist(filepath, 'file')
        delete(filepath);
    end
end
