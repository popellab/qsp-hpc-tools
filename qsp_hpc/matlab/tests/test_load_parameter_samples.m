classdef test_load_parameter_samples < matlab.unittest.TestCase
    %TEST_LOAD_PARAMETER_SAMPLES Unit tests for load_parameter_samples_csv.m
    %
    % Run with:
    %   results = runtests('test_load_parameter_samples');

    properties (TestParameter)
        % Parameterized test data
    end

    properties
        FixturesDir
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
        end
    end

    methods (TestClassTeardown)
        function restorePath(testCase)
            cd(testCase.OriginalDir);
        end
    end

    methods (Test)
        function test_loads_multiple_parameters(testCase)
            % Test loading CSV with multiple parameters
            csvFile = fullfile(testCase.FixturesDir, 'sample_params.csv');

            params = load_parameter_samples_csv(csvFile);

            % Check structure fields exist
            testCase.verifyTrue(isfield(params, 'names'), 'Missing names field');
            testCase.verifyTrue(isfield(params, 'all'), 'Missing all field');

            % Check correct number of parameters
            testCase.verifyEqual(length(params.names), 3, 'Expected 3 parameters');

            % Check correct number of samples
            testCase.verifyEqual(size(params.all, 1), 5, 'Expected 5 samples');
            testCase.verifyEqual(size(params.all, 2), 3, 'Expected 3 columns');
        end

        function test_parameter_names_correct(testCase)
            % Test that parameter names are correctly extracted
            csvFile = fullfile(testCase.FixturesDir, 'sample_params.csv');

            params = load_parameter_samples_csv(csvFile);

            expectedNames = {'k_elimination', 'V_distribution', 'k_absorption'};
            testCase.verifyEqual(params.names(:)', expectedNames, ...
                'Parameter names do not match expected');
        end

        function test_parameter_values_correct(testCase)
            % Test that parameter values are correctly loaded
            csvFile = fullfile(testCase.FixturesDir, 'sample_params.csv');

            params = load_parameter_samples_csv(csvFile);

            % Check first row values
            testCase.verifyEqual(params.all(1, 1), 0.15, 'AbsTol', 1e-10);
            testCase.verifyEqual(params.all(1, 2), 45.2, 'AbsTol', 1e-10);
            testCase.verifyEqual(params.all(1, 3), 1.2, 'AbsTol', 1e-10);

            % Check last row values
            testCase.verifyEqual(params.all(5, 1), 0.14, 'AbsTol', 1e-10);
            testCase.verifyEqual(params.all(5, 2), 41.9, 'AbsTol', 1e-10);
            testCase.verifyEqual(params.all(5, 3), 1.3, 'AbsTol', 1e-10);
        end

        function test_screenname_fields_created(testCase)
            % Test that ScreenName fields are created for each parameter
            csvFile = fullfile(testCase.FixturesDir, 'sample_params.csv');

            params = load_parameter_samples_csv(csvFile);

            % Each parameter should have a struct with ScreenName
            testCase.verifyTrue(isfield(params, 'k_elimination'));
            testCase.verifyTrue(isfield(params.k_elimination, 'ScreenName'));
            testCase.verifyEqual(params.k_elimination.ScreenName, 'k_elimination');

            testCase.verifyTrue(isfield(params, 'V_distribution'));
            testCase.verifyEqual(params.V_distribution.ScreenName, 'V_distribution');
        end

        function test_single_parameter(testCase)
            % Test loading CSV with single parameter
            csvFile = fullfile(testCase.FixturesDir, 'sample_params_single.csv');

            params = load_parameter_samples_csv(csvFile);

            testCase.verifyEqual(length(params.names), 1);
            testCase.verifyEqual(size(params.all, 1), 3);
            testCase.verifyEqual(size(params.all, 2), 1);
            testCase.verifyEqual(params.names{1}, 'k_elimination');
        end

        function test_missing_file_throws_error(testCase)
            % Test that missing file throws appropriate error
            csvFile = fullfile(testCase.FixturesDir, 'nonexistent.csv');

            testCase.verifyError(@() load_parameter_samples_csv(csvFile), ...
                ?MException);
        end

        function test_names_is_column_vector(testCase)
            % Test that names is returned as column vector
            csvFile = fullfile(testCase.FixturesDir, 'sample_params.csv');

            params = load_parameter_samples_csv(csvFile);

            testCase.verifyEqual(size(params.names, 2), 1, ...
                'names should be a column vector');
        end
    end

    methods (Test, TestTags = {'TempFile'})
        function test_handles_numeric_precision(testCase)
            % Test handling of high-precision numeric values
            import matlab.unittest.fixtures.TemporaryFolderFixture

            tempFixture = testCase.applyFixture(TemporaryFolderFixture);
            csvFile = fullfile(tempFixture.Folder, 'precision_test.csv');

            % Write CSV with high precision values
            fid = fopen(csvFile, 'w');
            fprintf(fid, 'param1,param2\n');
            fprintf(fid, '1.23456789012345,9.87654321098765\n');
            fclose(fid);

            params = load_parameter_samples_csv(csvFile);

            % Verify reasonable precision is maintained
            testCase.verifyEqual(params.all(1, 1), 1.23456789012345, ...
                'RelTol', 1e-10);
        end

        function test_handles_scientific_notation(testCase)
            % Test handling of scientific notation in CSV
            import matlab.unittest.fixtures.TemporaryFolderFixture

            tempFixture = testCase.applyFixture(TemporaryFolderFixture);
            csvFile = fullfile(tempFixture.Folder, 'scientific_test.csv');

            % Write CSV with scientific notation
            fid = fopen(csvFile, 'w');
            fprintf(fid, 'k_fast,k_slow\n');
            fprintf(fid, '1.5e-3,2.0e6\n');
            fprintf(fid, '3.2e-4,1.8e7\n');
            fclose(fid);

            params = load_parameter_samples_csv(csvFile);

            testCase.verifyEqual(params.all(1, 1), 1.5e-3, 'RelTol', 1e-10);
            testCase.verifyEqual(params.all(1, 2), 2.0e6, 'RelTol', 1e-10);
            testCase.verifyEqual(params.all(2, 1), 3.2e-4, 'RelTol', 1e-10);
        end
    end
end
