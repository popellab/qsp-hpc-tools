function params = load_parameter_samples_csv(csv_file)
% load_parameter_samples_csv - Load parameter samples from CSV file
%
% Reads pre-generated parameter samples from a CSV file where:
%   - First row contains parameter names
%   - Subsequent rows contain parameter values (one sample per row)
%
% This function is used when parameter samples are generated externally
% (e.g., by Python SBI workflow) and need to be loaded into MATLAB format.
%
% Input:
%   csv_file - Path to CSV file with parameter samples
%              Format: header row with parameter names, data rows with samples
%
% Output:
%   params - Parameter structure compatible with batch_execute()
%            Contains:
%            - names: Cell array of parameter names
%            - all: NxM matrix of parameter samples (N samples, M parameters)
%            - For each parameter: ScreenName
%
% Example:
%   % CSV format:
%   %   param1,param2,param3
%   %   0.5,1.2,3.4
%   %   0.7,1.1,3.8
%   %   ...
%
%   params = load_parameter_samples_csv('my_samples.csv');
%   % params.names = {'param1', 'param2', 'param3'}
%   % params.all = [N x 3] matrix of samples

fprintf('Reading parameter samples from: %s\n', csv_file);

% Read CSV file
opts = detectImportOptions(csv_file);

% Read as table
data_table = readtable(csv_file, opts);

% Peel off sample_index column if present (added by Python staging so MATLAB
% can thread it through to parquet for downstream cross-scenario alignment).
all_names = data_table.Properties.VariableNames;
if ~isempty(all_names) && strcmp(all_names{1}, 'sample_index')
    params.sample_indices = int64(data_table.sample_index);
    param_names = all_names(2:end);
    data_matrix = table2array(data_table(:, 2:end));
else
    params.sample_indices = int64((1:height(data_table))' - 1);
    param_names = all_names;
    data_matrix = table2array(data_table);
end
n_params = length(param_names);
n_samples = height(data_table);

fprintf('  Parameters: %d\n', n_params);
fprintf('  Samples: %d\n', n_samples);

% Initialize params structure
params.names = param_names(:);  % Column vector

% Convert table to matrix
params.all = data_matrix;

% Validate dimensions
if size(params.all, 2) ~= n_params
    error('Mismatch between number of parameters (%d) and columns (%d)', ...
        n_params, size(params.all, 2));
end

% Add ScreenName for each parameter (required for batch_execute)
for i = 1:n_params
    pname = param_names{i};
    params.(pname).ScreenName = pname;
end

fprintf('✅ Loaded %d samples for %d parameters\n', n_samples, n_params);

end
