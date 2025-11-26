function postproc_save_outputs(output_dir, array_idx, outputs)
%POSTPROC_SAVE_OUTPUTS Save postprocessing outputs to CSV files
%
% Generic function for saving postprocessing outputs (metrics, test statistics, etc.)
% to CSV files for easy download and aggregation.
%
% Inputs:
%   output_dir  - Directory to save output files
%   array_idx   - Chunk index (for naming files)
%   outputs     - Struct with postprocessing outputs, each field should have:
%                 .data   - Matrix of values (n_patients x n_outputs)
%                 .names  - Cell array of output names
%                 .type   - Output type ('metrics', 'test_stats', etc.) for filename
%
% Example:
%   outputs = struct();
%   outputs.test_stats = struct('data', test_stats_data, 'names', test_stat_ids, 'type', 'test_stats');
%   postproc_save_outputs(output_dir, 0, outputs);

output_types = fieldnames(outputs);

for i = 1:length(output_types)
    output_type = output_types{i};
    output = outputs.(output_type);

    % Skip if empty
    if isempty(output.data)
        continue;
    end

    % Save data CSV
    csv_filename = sprintf('chunk_%03d_%s.csv', array_idx, output.type);
    csv_file = fullfile(output_dir, csv_filename);
    writematrix(output.data, csv_file);

    % Save header (only for chunk 0)
    if array_idx == 0
        header_filename = sprintf('%s_header.txt', output.type);
        header_file = fullfile(output_dir, header_filename);
        fid = fopen(header_file, 'w');
        fprintf(fid, '%s\n', strjoin(output.names, ','));
        fclose(fid);
    end

    fprintf('   %s saved to: %s\n', output.type, csv_filename);
end

end
