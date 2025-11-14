function species_data = extract_all_species_arrays(chunk_results, model)
%EXTRACT_ALL_SPECIES_ARRAYS Extract time and all species from simulation results
%
% This function extracts full simulation outputs (time and all species)
% from SimBiology simulation results for saving to persistent storage.
%
% Inputs:
%   chunk_results  - Struct array with .simData fields (SimData objects)
%   model          - SimBiology model object (for species names)
%
% Outputs:
%   species_data   - Struct with fields:
%                    .n_sims         - Number of simulations
%                    .n_timepoints   - Number of time points per simulation
%                    .n_species      - Number of species
%                    .species_names  - Cell array of species names (1 x n_species)
%                    .time_arrays    - Cell array of time vectors (n_sims x 1)
%                    .species_arrays - Cell array of species matrices (n_sims x n_species)
%                    .status         - Status vector (n_sims x 1): 1=success, 0=failed IC, -1=failed sim
%
% Example:
%   species_data = extract_all_species_arrays(chunk_results, model);

n_sims = length(chunk_results);

% Get all species names from model (with compartment prefix for uniqueness)
all_species = sbioselect(model, 'Type', 'species');
species_names = cell(1, length(all_species));
for i = 1:length(all_species)
    % Get fully qualified name: compartment.species (e.g., V_T.C1)
    % This is critical for species that exist in multiple compartments
    species_obj = all_species(i);
    compartment_name = species_obj.Parent.Name;
    species_name = species_obj.Name;
    species_names{i} = [compartment_name '.' species_name];
end
n_species = length(species_names);

fprintf('   Extracting %d species from %d simulations...\n', n_species, n_sims);

% Initialize output arrays
time_arrays = cell(n_sims, 1);
species_arrays = cell(n_sims, n_species);
status = zeros(n_sims, 1);

% Extract data from each simulation
for i = 1:n_sims
    simdata = chunk_results(i).simData;

    if isempty(simdata)
        % Failed simulation - store empty arrays
        status(i) = -1;
        time_arrays{i} = [];
        for j = 1:n_species
            species_arrays{i, j} = [];
        end
    else
        % Successful simulation - extract time and species
        status(i) = 1;
        time_arrays{i} = simdata.Time;

        % Extract each species
        for j = 1:n_species
            species_name = species_names{j};
            try
                [~, data, ~] = selectbyname(simdata, species_name);
                species_arrays{i, j} = data;
            catch
                % Species not found or error - store NaN array
                species_arrays{i, j} = NaN(size(simdata.Time));
                fprintf('     ⚠️  Warning: Could not extract species %s for simulation %d\n', ...
                    species_name, i);
            end
        end
    end
end

% Package output
species_data = struct();
species_data.n_sims = n_sims;
species_data.n_timepoints = length(time_arrays{1});  % Assumes all sims have same timepoints
species_data.n_species = n_species;
species_data.species_names = species_names;
species_data.time_arrays = time_arrays;
species_data.species_arrays = species_arrays;
species_data.status = status;

n_success = sum(status == 1);
fprintf('   Extracted species data: %d/%d successful simulations\n', n_success, n_sims);

end
