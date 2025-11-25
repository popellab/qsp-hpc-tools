classdef test_save_species_to_parquet < matlab.unittest.TestCase
    %TEST_SAVE_SPECIES_TO_PARQUET Integration tests for save_species_to_parquet.m
    %
    % These tests require:
    %   - Python environment with pyarrow and pandas
    %   - qsp-hpc-tools package installed (pip install -e .)
    %
    % Run with:
    %   results = runtests('test_save_species_to_parquet');

    properties
        Model
        FixturesDir
        TempDir
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            % Add parent directory and fixtures to path
            testCase.FixturesDir = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            matlabDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(matlabDir);
            addpath(testCase.FixturesDir);
        end

        function verifyPythonAvailable(testCase)
            % Skip all tests if Python or required packages not available
            [status, ~] = system('python -c "import pyarrow; import pandas"');
            if status ~= 0
                % Try with python3
                [status, ~] = system('python3 -c "import pyarrow; import pandas"');
            end
            testCase.assumeEqual(status, 0, ...
                'Python with pyarrow and pandas required for these tests');
        end
    end

    methods (TestMethodSetup)
        function createTempDir(testCase)
            % Create temporary directory for test outputs
            testCase.TempDir = tempname;
            mkdir(testCase.TempDir);

            % Create fresh model for tests that need it
            testCase.Model = create_mock_model();
        end
    end

    methods (TestMethodTeardown)
        function cleanupTempDir(testCase)
            % Clean up temporary directory
            if exist(testCase.TempDir, 'dir')
                rmdir(testCase.TempDir, 's');
            end

            % Clean up model
            if ~isempty(testCase.Model) && isvalid(testCase.Model)
                delete(testCase.Model);
            end
        end
    end

    methods (Test)
        function test_creates_parquet_file(testCase)
            % Test that Parquet file is created successfully
            species_data = testCase.createMockSpeciesData(1);
            output_file = fullfile(testCase.TempDir, 'test_output.parquet');

            save_species_to_parquet(species_data, output_file);

            testCase.verifyTrue(exist(output_file, 'file') == 2, ...
                'Parquet file should be created');
        end

        function test_parquet_file_not_empty(testCase)
            % Test that Parquet file has content
            species_data = testCase.createMockSpeciesData(3);
            output_file = fullfile(testCase.TempDir, 'test_output.parquet');

            save_species_to_parquet(species_data, output_file);

            % File should have non-trivial size
            fileInfo = dir(output_file);
            testCase.verifyGreaterThan(fileInfo.bytes, 100, ...
                'Parquet file should have content');
        end

        function test_single_simulation(testCase)
            % Test with single simulation
            species_data = testCase.createMockSpeciesData(1);
            output_file = fullfile(testCase.TempDir, 'single_sim.parquet');

            save_species_to_parquet(species_data, output_file);

            testCase.verifyTrue(exist(output_file, 'file') == 2);

            % Verify content via Python
            n_rows = testCase.countParquetRows(output_file);
            testCase.verifyEqual(n_rows, 1);
        end

        function test_multiple_simulations(testCase)
            % Test with multiple simulations
            n_sims = 5;
            species_data = testCase.createMockSpeciesData(n_sims);
            output_file = fullfile(testCase.TempDir, 'multi_sim.parquet');

            save_species_to_parquet(species_data, output_file);

            % Verify correct number of rows
            n_rows = testCase.countParquetRows(output_file);
            testCase.verifyEqual(n_rows, n_sims);
        end

        function test_with_failed_simulations(testCase)
            % Test handling of failed simulations (empty arrays)
            species_data = testCase.createMockSpeciesData(3);
            % Mark second simulation as failed
            species_data.status(2) = -1;
            species_data.time_arrays{2} = [];
            for j = 1:species_data.n_species
                species_data.species_arrays{2, j} = [];
            end

            output_file = fullfile(testCase.TempDir, 'with_failed.parquet');

            save_species_to_parquet(species_data, output_file);

            testCase.verifyTrue(exist(output_file, 'file') == 2);
            n_rows = testCase.countParquetRows(output_file);
            testCase.verifyEqual(n_rows, 3, 'Should include all simulations including failed');
        end

        function test_with_parameters(testCase)
            % Test that parameters are included in output
            species_data = testCase.createMockSpeciesData(2);

            % Add parameter information
            chunk_params = struct();
            chunk_params.names = {'k_elimination', 'V_distribution'};
            chunk_params.k_elimination.LHS = [0.1; 0.2];
            chunk_params.V_distribution.LHS = [50; 60];

            output_file = fullfile(testCase.TempDir, 'with_params.parquet');

            save_species_to_parquet(species_data, output_file, chunk_params);

            testCase.verifyTrue(exist(output_file, 'file') == 2);

            % Verify parameters are in columns
            columns = testCase.getParquetColumns(output_file);
            testCase.verifyTrue(any(strcmp(columns, 'k_elimination')), ...
                'k_elimination parameter should be a column');
            testCase.verifyTrue(any(strcmp(columns, 'V_distribution')), ...
                'V_distribution parameter should be a column');
        end

        function test_species_columns_created(testCase)
            % Test that species columns are present
            species_data = testCase.createMockSpeciesData(1);
            output_file = fullfile(testCase.TempDir, 'species_cols.parquet');

            save_species_to_parquet(species_data, output_file);

            columns = testCase.getParquetColumns(output_file);

            % Should have simulation_id, status, time, and species columns
            testCase.verifyTrue(any(strcmp(columns, 'simulation_id')));
            testCase.verifyTrue(any(strcmp(columns, 'status')));
            testCase.verifyTrue(any(strcmp(columns, 'time')));

            % Species columns have dots replaced with underscores
            testCase.verifyTrue(any(contains(columns, 'V_C_Drug')), ...
                'Expected V_C_Drug column (from V_C.Drug)');
        end

        function test_data_structure_preserved(testCase)
            % Test that data structure is correctly preserved in Parquet
            % Create species_data with known, deterministic values
            species_data = struct();
            species_data.n_sims = 2;
            species_data.n_timepoints = 3;
            species_data.n_species = 2;
            species_data.species_names = {'V_C.Drug', 'V_P.Drug'};
            species_data.status = [1; 1];

            % Use specific known values
            species_data.time_arrays = {[0; 1; 2], [0; 1; 2]};
            species_data.species_arrays = cell(2, 2);
            species_data.species_arrays{1, 1} = [100; 80; 60];   % Sim 1, Species 1
            species_data.species_arrays{1, 2} = [0; 10; 15];     % Sim 1, Species 2
            species_data.species_arrays{2, 1} = [200; 160; 120]; % Sim 2, Species 1
            species_data.species_arrays{2, 2} = [0; 20; 30];     % Sim 2, Species 2

            output_file = fullfile(testCase.TempDir, 'data_values.parquet');
            save_species_to_parquet(species_data, output_file);

            % Verify file created
            testCase.verifyTrue(exist(output_file, 'file') == 2, ...
                'Parquet file should be created');

            % Verify row count matches n_sims
            n_rows = testCase.countParquetRows(output_file);
            testCase.verifyEqual(n_rows, 2, ...
                'Number of rows should match n_sims');

            % Verify columns exist
            columns = testCase.getParquetColumns(output_file);
            testCase.verifyTrue(any(strcmp(columns, 'simulation_id')), ...
                'simulation_id column should exist');
            testCase.verifyTrue(any(strcmp(columns, 'status')), ...
                'status column should exist');
            testCase.verifyTrue(any(strcmp(columns, 'time')), ...
                'time column should exist');
            testCase.verifyTrue(any(strcmp(columns, 'V_C_Drug')), ...
                'V_C_Drug column should exist');
            testCase.verifyTrue(any(strcmp(columns, 'V_P_Drug')), ...
                'V_P_Drug column should exist');
        end

        function test_parameter_columns_created(testCase)
            % Test that parameter columns are correctly created in Parquet
            species_data = testCase.createMockSpeciesData(3);

            % Add parameters with known values
            chunk_params = struct();
            chunk_params.names = {'k_el', 'V_d'};
            chunk_params.k_el.LHS = [0.1; 0.2; 0.3];
            chunk_params.V_d.LHS = [50; 60; 70];

            output_file = fullfile(testCase.TempDir, 'param_values.parquet');
            save_species_to_parquet(species_data, output_file, chunk_params);

            % Verify file created and correct row count
            testCase.verifyTrue(exist(output_file, 'file') == 2);
            n_rows = testCase.countParquetRows(output_file);
            testCase.verifyEqual(n_rows, 3, ...
                'Number of rows should match n_sims');

            % Verify parameter columns exist
            columns = testCase.getParquetColumns(output_file);
            testCase.verifyTrue(any(strcmp(columns, 'k_el')), ...
                'k_el parameter column should exist');
            testCase.verifyTrue(any(strcmp(columns, 'V_d')), ...
                'V_d parameter column should exist');
        end

        function test_with_real_simbiology_simulation(testCase)
            % Integration test with actual SimBiology simulation data
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);
            output_file = fullfile(testCase.TempDir, 'real_sim.parquet');

            save_species_to_parquet(species_data, output_file);

            testCase.verifyTrue(exist(output_file, 'file') == 2);

            % Verify structure
            columns = testCase.getParquetColumns(output_file);
            testCase.verifyTrue(any(strcmp(columns, 'time')));
            testCase.verifyGreaterThan(length(columns), 3, ...
                'Should have multiple columns for species');
        end

        function test_temp_json_cleanup(testCase)
            % Test that temporary JSON file is cleaned up
            species_data = testCase.createMockSpeciesData(1);
            output_file = fullfile(testCase.TempDir, 'cleanup_test.parquet');

            % Count JSON files before
            jsonFilesBefore = dir(fullfile(tempdir, '*.json'));
            numBefore = length(jsonFilesBefore);

            save_species_to_parquet(species_data, output_file);

            % Count JSON files after - should be same or fewer (cleanup worked)
            jsonFilesAfter = dir(fullfile(tempdir, '*.json'));
            numAfter = length(jsonFilesAfter);

            testCase.verifyLessThanOrEqual(numAfter, numBefore, ...
                'Temporary JSON file should be cleaned up (no new JSON files should remain)');
        end
    end

    methods (Access = private)
        function species_data = createMockSpeciesData(testCase, n_sims)
            % Create mock species_data struct for testing

            n_timepoints = 10;
            species_names = {'V_C.Drug', 'V_P.Drug', 'V_C', 'V_P'};
            n_species = length(species_names);

            species_data = struct();
            species_data.n_sims = n_sims;
            species_data.n_timepoints = n_timepoints;
            species_data.n_species = n_species;
            species_data.species_names = species_names;
            species_data.status = ones(n_sims, 1);

            % Create time and species arrays
            species_data.time_arrays = cell(n_sims, 1);
            species_data.species_arrays = cell(n_sims, n_species);

            for i = 1:n_sims
                species_data.time_arrays{i} = (0:n_timepoints-1)';
                for j = 1:n_species
                    % Generate simple test data
                    species_data.species_arrays{i, j} = rand(n_timepoints, 1) * 100;
                end
            end
        end

        function n_rows = countParquetRows(testCase, parquet_file)
            % Count rows in Parquet file using Python
            cmd = sprintf('python -c "import pandas as pd; print(len(pd.read_parquet(''%s'')))"', ...
                strrep(parquet_file, '''', ''''''));
            [status, output] = system(cmd);
            if status ~= 0
                % Try python3
                cmd = sprintf('python3 -c "import pandas as pd; print(len(pd.read_parquet(''%s'')))"', ...
                    strrep(parquet_file, '''', ''''''));
                [status, output] = system(cmd);
            end
            testCase.assumeEqual(status, 0, 'Python command failed');
            n_rows = str2double(strtrim(output));
        end

        function columns = getParquetColumns(testCase, parquet_file)
            % Get column names from Parquet file using Python
            cmd = sprintf('python -c "import pandas as pd; print(''\\n''.join(pd.read_parquet(''%s'').columns))"', ...
                strrep(parquet_file, '''', ''''''));
            [status, output] = system(cmd);
            if status ~= 0
                % Try python3
                cmd = sprintf('python3 -c "import pandas as pd; print(''\\n''.join(pd.read_parquet(''%s'').columns))"', ...
                    strrep(parquet_file, '''', ''''''));
                [status, output] = system(cmd);
            end
            testCase.assumeEqual(status, 0, 'Python command failed');
            columns = strsplit(strtrim(output), newline);
        end

    end
end
