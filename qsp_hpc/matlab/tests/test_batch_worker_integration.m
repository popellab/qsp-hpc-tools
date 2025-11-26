classdef test_batch_worker_integration < matlab.unittest.TestCase
    %TEST_BATCH_WORKER_INTEGRATION Integration tests for batch_worker components
    %
    % Tests the postprocessing pipeline used by batch_worker.m with mocked
    % simulation data to verify correct behavior without requiring actual
    % SimBiology simulations.
    %
    % Run with:
    %   results = runtests('test_batch_worker_integration');

    properties
        FixturesDir
        TempDir
        OriginalDir
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            % Add parent directory (matlab/) to path for function access
            testCase.OriginalDir = pwd;
            testCase.FixturesDir = fullfile(fileparts(mfilename('fullpath')), 'fixtures');

            % Add the matlab directory to path
            matlabDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(matlabDir);
            addpath(testCase.FixturesDir);
        end
    end

    methods (TestMethodSetup)
        function createTempDir(testCase)
            % Create fresh temp directory for each test
            testCase.TempDir = tempname;
            mkdir(testCase.TempDir);
        end
    end

    methods (TestMethodTeardown)
        function cleanupTempDir(testCase)
            % Remove temp directory after each test
            if exist(testCase.TempDir, 'dir')
                rmdir(testCase.TempDir, 's');
            end
        end
    end

    methods (TestClassTeardown)
        function restorePath(testCase)
            cd(testCase.OriginalDir);
        end
    end

    %% Helper Methods
    methods (Access = private)
        function chunk_results = createMockChunkResults(testCase, n_patients)
            %CREATEMOCKCHUNKRESULTS Create real simulation results for testing
            %
            % Creates a struct array with actual SimBiology SimData objects
            % from running the mock model with different parameters.

            chunk_results = struct('simData', cell(1, n_patients));

            % Create and configure the mock model
            model = create_mock_model();

            % Configure simulation
            cs = getconfigset(model, 'active');
            cs.SolverType = 'ode15s';
            cs.StopTime = 24;
            set(cs.SolverOptions, 'AbsoluteTolerance', 1e-9);
            set(cs.SolverOptions, 'RelativeTolerance', 1e-6);

            for i = 1:n_patients
                % Create variant with different elimination rate for each patient
                variant = addvariant(model, sprintf('Patient%d', i));
                k_el_value = 0.05 + 0.02 * i;  % Varies by patient
                addcontent(variant, {'parameter', 'k_elimination', 'Value', k_el_value});

                % Run simulation
                simData = sbiosimulate(model, variant);
                chunk_results(i).simData = simData;

                % Clean up variant
                delete(variant);
            end

            % Clean up model
            delete(model);
        end

        function model = createMockModel(~)
            %CREATEMOCKMODEL Create SimBiology model for testing using fixture
            model = create_mock_model();
        end
    end

    %% Tests for postproc_extract_test_statistics
    methods (Test)
        function test_extract_test_stats_with_mock_data(testCase)
            % Test that postproc_extract_test_statistics correctly extracts
            % statistics from mock simulation data

            % Setup: copy test stats CSV and function files to temp dir
            copyfile(fullfile(testCase.FixturesDir, 'integration_test_stats.csv'), ...
                     fullfile(testCase.TempDir, 'test_stats.csv'));
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_peak_drug.m'), testCase.TempDir);
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_auc_drug.m'), testCase.TempDir);
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_trough_drug.m'), testCase.TempDir);

            % Create mock simulation results
            n_patients = 3;
            chunk_results = testCase.createMockChunkResults(n_patients);

            % Create mock model
            model = testCase.createMockModel();

            % Run extraction
            config = struct();
            [test_stats_data, test_stat_ids, metadata] = ...
                postproc_extract_test_statistics(chunk_results, model, config, testCase.TempDir);

            % Verify outputs
            testCase.verifyEqual(size(test_stats_data, 1), n_patients, ...
                'Should have one row per patient');
            testCase.verifyEqual(size(test_stats_data, 2), 3, ...
                'Should have 3 test statistics');
            testCase.verifyEqual(length(test_stat_ids), 3, ...
                'Should have 3 test statistic IDs');

            % Verify test stat IDs
            testCase.verifyEqual(test_stat_ids{1}, 'peak_drug');
            testCase.verifyEqual(test_stat_ids{2}, 'auc_drug');
            testCase.verifyEqual(test_stat_ids{3}, 'trough_drug');

            % Verify metadata
            testCase.verifyTrue(isfield(metadata, 'units'));
            testCase.verifyTrue(isfield(metadata, 'expected_mean'));
            testCase.verifyTrue(isfield(metadata, 'expected_variance'));

            % Verify no NaN values (all extractions should succeed)
            testCase.verifyFalse(any(isnan(test_stats_data(:))), ...
                'All test statistics should be successfully extracted');

            % Clean up
            delete(model);
        end

        function test_extract_test_stats_handles_missing_function(testCase)
            % Test graceful handling when test statistic function is missing

            % Create a test stats CSV that references functions that don't exist anywhere
            test_stats_csv = fullfile(testCase.TempDir, 'test_stats.csv');
            fid = fopen(test_stats_csv, 'w');
            fprintf(fid, 'test_statistic_id,mean,variance,units,required_species,model_output_code\n');
            fprintf(fid, 'nonexistent_stat1,100.0,25.0,ng/mL,V_C.Drug,""\n');
            fprintf(fid, 'nonexistent_stat2,200.0,50.0,ng/mL,V_C.Drug,""\n');
            fclose(fid);

            % Create mock simulation results
            chunk_results = testCase.createMockChunkResults(2);
            model = testCase.createMockModel();

            % Run extraction - should not error, just return NaN for missing functions
            config = struct();
            [test_stats_data, test_stat_ids, ~] = ...
                postproc_extract_test_statistics(chunk_results, model, config, testCase.TempDir);

            % All values should be NaN since functions are missing
            testCase.verifyTrue(all(isnan(test_stats_data(:))), ...
                'All values should be NaN when functions are missing');
            testCase.verifyEqual(length(test_stat_ids), 2, ...
                'Should have 2 test statistic IDs');

            delete(model);
        end

        function test_extract_test_stats_no_csv(testCase)
            % Test behavior when no test_stats.csv exists

            chunk_results = testCase.createMockChunkResults(2);
            model = testCase.createMockModel();

            % Run extraction with empty temp dir (no CSV)
            config = struct();
            [test_stats_data, test_stat_ids, metadata] = ...
                postproc_extract_test_statistics(chunk_results, model, config, testCase.TempDir);

            % Should return empty outputs
            testCase.verifyTrue(isempty(test_stats_data));
            testCase.verifyTrue(isempty(test_stat_ids));
            testCase.verifyTrue(isempty(fieldnames(metadata)));

            delete(model);
        end
    end

    %% Tests for postproc_save_outputs
    methods (Test)
        function test_save_outputs_creates_csv(testCase)
            % Test that postproc_save_outputs creates correct CSV files

            % Create mock outputs structure
            outputs = struct();
            outputs.test_stats = struct(...
                'data', [1.0 2.0 3.0; 4.0 5.0 6.0], ...
                'names', {{'stat1', 'stat2', 'stat3'}}, ...
                'type', 'test_stats');

            % Save outputs
            array_idx = 0;
            postproc_save_outputs(testCase.TempDir, array_idx, outputs);

            % Verify CSV file exists
            csv_file = fullfile(testCase.TempDir, 'chunk_000_test_stats.csv');
            testCase.verifyTrue(exist(csv_file, 'file') == 2, ...
                'CSV file should be created');

            % Verify data can be read back
            data = readmatrix(csv_file);
            testCase.verifyEqual(size(data), [2 3], 'Data dimensions should match');
            testCase.verifyEqual(data(1, 1), 1.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(data(2, 3), 6.0, 'AbsTol', 1e-10);
        end

        function test_save_outputs_creates_header_for_chunk_zero(testCase)
            % Test that header file is created only for chunk 0

            outputs = struct();
            outputs.test_stats = struct(...
                'data', [1.0 2.0], ...
                'names', {{'stat1', 'stat2'}}, ...
                'type', 'test_stats');

            % Save for chunk 0
            postproc_save_outputs(testCase.TempDir, 0, outputs);

            % Verify header file exists
            header_file = fullfile(testCase.TempDir, 'test_stats_header.txt');
            testCase.verifyTrue(exist(header_file, 'file') == 2, ...
                'Header file should be created for chunk 0');

            % Verify header content
            fid = fopen(header_file, 'r');
            header_content = fgetl(fid);
            fclose(fid);
            testCase.verifyEqual(header_content, 'stat1,stat2');
        end

        function test_save_outputs_no_header_for_chunk_nonzero(testCase)
            % Test that header file is NOT created for chunk > 0

            outputs = struct();
            outputs.test_stats = struct(...
                'data', [1.0 2.0], ...
                'names', {{'stat1', 'stat2'}}, ...
                'type', 'test_stats');

            % Save for chunk 5
            postproc_save_outputs(testCase.TempDir, 5, outputs);

            % Verify header file does NOT exist
            header_file = fullfile(testCase.TempDir, 'test_stats_header.txt');
            testCase.verifyFalse(exist(header_file, 'file') == 2, ...
                'Header file should NOT be created for chunk > 0');

            % But CSV should still exist
            csv_file = fullfile(testCase.TempDir, 'chunk_005_test_stats.csv');
            testCase.verifyTrue(exist(csv_file, 'file') == 2, ...
                'CSV file should still be created');
        end

        function test_save_outputs_skips_empty_data(testCase)
            % Test that empty data is skipped

            outputs = struct();
            outputs.test_stats = struct(...
                'data', [], ...
                'names', {{}}, ...
                'type', 'test_stats');

            % Save outputs
            postproc_save_outputs(testCase.TempDir, 0, outputs);

            % Verify no CSV file created
            csv_file = fullfile(testCase.TempDir, 'chunk_000_test_stats.csv');
            testCase.verifyFalse(exist(csv_file, 'file') == 2, ...
                'CSV file should NOT be created for empty data');
        end

        function test_save_outputs_multiple_types(testCase)
            % Test saving multiple output types

            outputs = struct();
            outputs.test_stats = struct(...
                'data', [1.0 2.0], ...
                'names', {{'stat1', 'stat2'}}, ...
                'type', 'test_stats');
            outputs.metrics = struct(...
                'data', [10.0 20.0 30.0], ...
                'names', {{'metric1', 'metric2', 'metric3'}}, ...
                'type', 'metrics');

            % Save outputs
            postproc_save_outputs(testCase.TempDir, 0, outputs);

            % Verify both CSV files exist
            testCase.verifyTrue(exist(fullfile(testCase.TempDir, 'chunk_000_test_stats.csv'), 'file') == 2);
            testCase.verifyTrue(exist(fullfile(testCase.TempDir, 'chunk_000_metrics.csv'), 'file') == 2);
        end
    end

    %% Integration tests combining multiple components
    methods (Test)
        function test_full_postprocessing_pipeline(testCase)
            % End-to-end test of the postprocessing pipeline

            % Setup input directory with test stats
            input_dir = fullfile(testCase.TempDir, 'input');
            output_dir = fullfile(testCase.TempDir, 'output');
            mkdir(input_dir);
            mkdir(output_dir);

            % Copy test stats files
            copyfile(fullfile(testCase.FixturesDir, 'integration_test_stats.csv'), ...
                     fullfile(input_dir, 'test_stats.csv'));
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_peak_drug.m'), input_dir);
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_auc_drug.m'), input_dir);
            copyfile(fullfile(testCase.FixturesDir, 'test_stat_trough_drug.m'), input_dir);

            % Create mock simulation results
            n_patients = 5;
            chunk_results = testCase.createMockChunkResults(n_patients);
            model = testCase.createMockModel();

            % Step 1: Extract test statistics
            config = struct();
            [test_stats_data, test_stat_ids, ~] = ...
                postproc_extract_test_statistics(chunk_results, model, config, input_dir);

            % Step 2: Build outputs structure
            outputs = struct();
            outputs.test_stats = struct(...
                'data', test_stats_data, ...
                'names', {test_stat_ids}, ...
                'type', 'test_stats');

            % Step 3: Save outputs
            array_idx = 0;
            postproc_save_outputs(output_dir, array_idx, outputs);

            % Verify final output
            csv_file = fullfile(output_dir, 'chunk_000_test_stats.csv');
            testCase.verifyTrue(exist(csv_file, 'file') == 2, ...
                'Final CSV should exist');

            % Load and verify data
            saved_data = readmatrix(csv_file);
            testCase.verifyEqual(size(saved_data), [n_patients, 3], ...
                'Saved data should have correct dimensions');

            % Peak should be positive
            testCase.verifyTrue(all(saved_data(:, 1) > 0), ...
                'Peak values should be positive');

            % AUC should be positive
            testCase.verifyTrue(all(saved_data(:, 2) > 0), ...
                'AUC values should be positive');

            % Trough should be less than peak
            testCase.verifyTrue(all(saved_data(:, 3) < saved_data(:, 1)), ...
                'Trough should be less than peak');

            delete(model);
        end
    end
end
