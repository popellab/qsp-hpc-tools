classdef test_extract_all_species_arrays < matlab.unittest.TestCase
    %TEST_EXTRACT_ALL_SPECIES_ARRAYS Unit tests for extract_all_species_arrays.m
    %
    % These tests use a minimal SimBiology model to verify species extraction.
    %
    % Run with:
    %   results = runtests('test_extract_all_species_arrays');

    properties
        Model
        FixturesDir
    end

    methods (TestClassSetup)
        function setupPath(testCase)
            % Add parent directory and fixtures to path
            testCase.FixturesDir = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            matlabDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(matlabDir);
            addpath(testCase.FixturesDir);
        end
    end

    methods (TestMethodSetup)
        function createModel(testCase)
            % Create fresh model for each test
            testCase.Model = create_mock_model();
        end
    end

    methods (TestMethodTeardown)
        function deleteModel(testCase)
            % Clean up model
            if ~isempty(testCase.Model)
                delete(testCase.Model);
            end
        end
    end

    methods (Test)
        function test_extracts_species_from_single_simulation(testCase)
            % Test extraction from a single successful simulation
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            testCase.verifyEqual(species_data.n_sims, 1);
            testCase.verifyGreaterThan(species_data.n_species, 0);
            testCase.verifyGreaterThan(species_data.n_timepoints, 0);
        end

        function test_species_names_include_compartment_prefix(testCase)
            % Test that species names are fully qualified (compartment.species)
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            % Should have compartment-prefixed species names
            hasQualifiedNames = any(contains(species_data.species_names, '.'));
            testCase.verifyTrue(hasQualifiedNames, ...
                'Species names should include compartment prefix (e.g., V_C.Drug)');

            % Check specific expected names
            testCase.verifyTrue(any(strcmp(species_data.species_names, 'V_C.Drug')), ...
                'Expected V_C.Drug in species names');
            testCase.verifyTrue(any(strcmp(species_data.species_names, 'V_P.Drug')), ...
                'Expected V_P.Drug in species names');
        end

        function test_includes_compartment_volumes(testCase)
            % Test that compartment volumes are included in output
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            % Compartment names should be in the list (without prefix)
            testCase.verifyTrue(any(strcmp(species_data.species_names, 'V_C')), ...
                'Expected V_C compartment in species names');
            testCase.verifyTrue(any(strcmp(species_data.species_names, 'V_P')), ...
                'Expected V_P compartment in species names');
        end

        function test_compartment_capacity_fallback(testCase)
            % Test that compartment volumes use model Capacity when not logged in simdata
            % This tests the fallback behavior when selectbyname returns empty for compartments
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            % Find compartment indices
            v_c_idx = find(strcmp(species_data.species_names, 'V_C'));
            v_p_idx = find(strcmp(species_data.species_names, 'V_P'));

            testCase.verifyNotEmpty(v_c_idx, 'V_C should be in species_names');
            testCase.verifyNotEmpty(v_p_idx, 'V_P should be in species_names');

            % Get compartment data
            v_c_data = species_data.species_arrays{1, v_c_idx};
            v_p_data = species_data.species_arrays{1, v_p_idx};

            % Verify correct length (should match timepoints)
            testCase.verifyEqual(length(v_c_data), species_data.n_timepoints, ...
                'V_C data should have correct number of timepoints');
            testCase.verifyEqual(length(v_p_data), species_data.n_timepoints, ...
                'V_P data should have correct number of timepoints');

            % Verify values match model Capacity (from create_mock_model: V_C=50, V_P=100)
            testCase.verifyEqual(unique(v_c_data), 50, ...
                'V_C values should equal model Capacity (50)');
            testCase.verifyEqual(unique(v_p_data), 100, ...
                'V_P values should equal model Capacity (100)');
        end

        function test_extracts_multiple_simulations(testCase)
            % Test extraction from multiple simulations
            n_sims = 3;
            chunk_results = struct('simData', cell(1, n_sims));

            for i = 1:n_sims
                chunk_results(i).simData = sbiosimulate(testCase.Model);
            end

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            testCase.verifyEqual(species_data.n_sims, n_sims);
            testCase.verifyEqual(length(species_data.time_arrays), n_sims);
            testCase.verifyEqual(size(species_data.species_arrays, 1), n_sims);
        end

        function test_handles_failed_simulation(testCase)
            % Test handling of failed simulations (empty simData)
            chunk_results(1).simData = sbiosimulate(testCase.Model);
            chunk_results(2).simData = [];  % Failed simulation
            chunk_results(3).simData = sbiosimulate(testCase.Model);

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            testCase.verifyEqual(species_data.n_sims, 3);

            % Check status vector
            testCase.verifyEqual(species_data.status(1), 1, 'First sim should be success');
            testCase.verifyEqual(species_data.status(2), -1, 'Second sim should be failed');
            testCase.verifyEqual(species_data.status(3), 1, 'Third sim should be success');

            % Failed simulation should have empty arrays
            testCase.verifyEmpty(species_data.time_arrays{2});
        end

        function test_time_arrays_are_correct(testCase)
            % Test that time arrays match simulation output
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            testCase.verifyEqual(species_data.time_arrays{1}, simData.Time, ...
                'Time array should match simulation time');
        end

        function test_species_data_dimensions(testCase)
            % Test that species data has correct dimensions
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            % species_arrays should be n_sims x n_species cell array
            testCase.verifyEqual(size(species_data.species_arrays, 1), 1);
            testCase.verifyEqual(size(species_data.species_arrays, 2), species_data.n_species);

            % Each species array should have length equal to timepoints
            for j = 1:species_data.n_species
                testCase.verifyEqual(length(species_data.species_arrays{1, j}), ...
                    species_data.n_timepoints, ...
                    sprintf('Species %d should have %d timepoints', j, species_data.n_timepoints));
            end
        end

        function test_output_structure_fields(testCase)
            % Test that output struct has all required fields
            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            requiredFields = {'n_sims', 'n_timepoints', 'n_species', ...
                'species_names', 'time_arrays', 'species_arrays', 'status'};

            for i = 1:length(requiredFields)
                testCase.verifyTrue(isfield(species_data, requiredFields{i}), ...
                    sprintf('Missing required field: %s', requiredFields{i}));
            end
        end

        function test_all_failed_simulations(testCase)
            % Test handling when all simulations failed
            chunk_results(1).simData = [];
            chunk_results(2).simData = [];

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            testCase.verifyEqual(species_data.n_sims, 2);
            testCase.verifyEqual(sum(species_data.status), -2, ...
                'All simulations should be marked as failed');
        end
    end

    methods (Test, TestTags = {'SimBiology'})
        function test_with_modified_parameters(testCase)
            % Test extraction with modified model parameters
            % This simulates what happens in actual batch processing

            % Modify a parameter
            k_el = sbioselect(testCase.Model, 'Type', 'parameter', 'Name', 'k_elimination');
            k_el.Value = 0.2;  % Double elimination rate

            simData = sbiosimulate(testCase.Model);
            chunk_results(1).simData = simData;

            species_data = extract_all_species_arrays(chunk_results, testCase.Model);

            % Should still extract correctly
            testCase.verifyEqual(species_data.n_sims, 1);
            testCase.verifyGreaterThan(species_data.n_species, 0);

            % Drug should decline faster with higher elimination
            drug_idx = find(strcmp(species_data.species_names, 'V_C.Drug'));
            drug_data = species_data.species_arrays{1, drug_idx};
            testCase.verifyLessThan(drug_data(end), drug_data(1), ...
                'Drug concentration should decrease over time');
        end
    end
end
