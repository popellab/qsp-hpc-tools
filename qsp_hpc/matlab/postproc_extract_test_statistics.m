function [test_stats_data, test_stat_ids, test_stats_metadata] = postproc_extract_test_statistics(chunk_results, model, config, input_dir)
%POSTPROC_EXTRACT_TEST_STATISTICS Extract test statistics from chunk simulation results
%
% Generic postprocessing function for extracting test statistics from simulation results.
% Reads test statistics CSV and calls uploaded test statistic functions.
%
% Inputs:
%   chunk_results  - Struct array of simulation results (each has .simData field)
%   model          - SimBiology model object
%   config         - Configuration struct (reserved for future use)
%   input_dir      - Directory containing test_stats.csv and function files
%
% Outputs:
%   test_stats_data     - Matrix of test statistic values (n_patients x n_test_stats)
%   test_stat_ids       - Cell array of test statistic IDs
%   test_stats_metadata - Struct with fields: units, expected_mean, expected_variance
%
% Example:
%   [data, ids, meta] = postproc_extract_test_statistics(chunk_results, model, struct(), input_dir);

test_stats_data = [];
test_stat_ids = {};
test_stats_metadata = struct();

% Look for test_stats.csv in input_dir
test_stats_csv = fullfile(input_dir, 'test_stats.csv');
if ~exist(test_stats_csv, 'file')
    return;  % No test statistics CSV provided
end

if ~exist(input_dir, 'dir')
    fprintf('   ⚠️  Input directory not found: %s\n', input_dir);
    return;
end

fprintf('   Extracting test statistics using uploaded functions...\n');

try
    % Read test statistics CSV
    test_stats_table = readtable(test_stats_csv, 'Delimiter', ',', 'ReadVariableNames', true);
    n_test_stats = height(test_stats_table);
    n_patients = length(chunk_results);

    fprintf('   Processing %d test statistics for %d patients...\n', n_test_stats, n_patients);

    % Initialize output
    test_stats_data = NaN(n_patients, n_test_stats);
    test_stat_ids = cellstr(test_stats_table.test_statistic_id);
    test_stats_metadata.units = cellstr(test_stats_table.units);
    test_stats_metadata.expected_mean = test_stats_table.mean;
    test_stats_metadata.expected_variance = test_stats_table.variance;

    % Get required species for each test statistic (if column exists)
    has_required_species = ismember('required_species', test_stats_table.Properties.VariableNames);
    if has_required_species
        required_species_list = cellstr(test_stats_table.required_species);
    else
        fprintf('   ⚠️  CSV missing required_species column - test statistics may fail\n');
        required_species_list = cell(n_test_stats, 1);
    end

    % Add input directory to path to find uploaded function files
    addpath(input_dir);

    % Process each test statistic using uploaded function files
    for j = 1:n_test_stats
        test_stat_id = test_stat_ids{j};
        function_name = sprintf('test_stat_%s', test_stat_id);

        % Check if function file exists
        if ~exist(function_name, 'file')
            fprintf('     ⚠️  Function not found: %s.m\n', function_name);
            continue;
        end

        % Parse required species (comma-separated list)
        required_species_str = required_species_list{j};
        use_species_extraction = has_required_species && ~isempty(required_species_str);

        if use_species_extraction
            required_species = strsplit(required_species_str, ',');
        end

        % Apply function to each patient
        for i = 1:n_patients
            try
                simdata = chunk_results(i).simData;
                if ~isempty(simdata)
                    if use_species_extraction
                        % Extract time and required species from simdata
                        time = simdata.Time;
                        species_data = {time};  % First arg is always time

                        % Extract each required species
                        for k = 1:length(required_species)
                            species_name = strtrim(required_species{k});
                            [~, data, ~] = selectbyname(simdata, species_name);
                            species_data{end+1} = data;
                        end

                        % Call function with extracted data
                        test_stat_value = feval(function_name, species_data{:});
                        test_stats_data(i, j) = test_stat_value;
                        fprintf('     Patient %d, %s: %.4e\n', i, test_stat_id, test_stat_value);
                    else
                        % Fallback: pass simdata directly (old behavior)
                        test_stat_value = feval(function_name, simdata);
                        test_stats_data(i, j) = test_stat_value;
                        fprintf('     Patient %d, %s: %.4e\n', i, test_stat_id, test_stat_value);
                    end
                end
            catch func_err
                fprintf('     ⚠️  Error computing %s for patient %d: %s\n', ...
                    test_stat_id, i, func_err.message);
            end
        end
    end

    % Remove input directory from path
    rmpath(input_dir);

    n_extracted = sum(~isnan(test_stats_data(:)));
    n_total = numel(test_stats_data);
    fprintf('   Test statistics extracted: %d/%d values (%.1f%%)\n', ...
        n_extracted, n_total, 100*n_extracted/n_total);

catch extract_err
    fprintf('   ⚠️  Test statistic extraction failed: %s\n', extract_err.message);
    if ~isempty(extract_err.stack)
        fprintf('      Error location: %s (line %d)\n', extract_err.stack(1).name, extract_err.stack(1).line);
    end
    test_stats_data = [];
    test_stat_ids = {};
    test_stats_metadata = struct();
end

end
