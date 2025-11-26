function species_data = extract_all_species_arrays(chunk_results, model)
%EXTRACT_ALL_SPECIES_ARRAYS Extract time, species, and compartments from simulation results
%
% This function extracts full simulation outputs (time, all species, and
% compartment volumes) from SimBiology simulation results for saving to
% persistent storage.
%
% Inputs:
%   chunk_results  - Struct array with .simData fields (SimData objects)
%   model          - SimBiology model object (for species names)
%
% Outputs:
%   species_data   - Struct with fields:
%                    .n_sims         - Number of simulations
%                    .n_timepoints   - Number of time points per simulation
%                    .n_species      - Number of species + compartments
%                    .species_names  - Cell array of names (1 x n_species)
%                    .time_arrays    - Cell array of time vectors (n_sims x 1)
%                    .species_arrays - Cell array of data matrices (n_sims x n_species)
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

% Also get compartment volumes (e.g., V_T, V_C, V_LN, V_P)
% These are needed for computing cell densities in test statistics
all_compartments = sbioselect(model, 'Type', 'compartment');
compartment_names = cell(1, length(all_compartments));
compartment_capacities = containers.Map();
for i = 1:length(all_compartments)
    compartment_names{i} = all_compartments(i).Name;
    compartment_capacities(all_compartments(i).Name) = all_compartments(i).Capacity;
end

% Combine species and compartment names
% Compartments come after species for backwards compatibility
species_names = [species_names, compartment_names];
n_species = length(species_names);
n_compartments = length(compartment_names);

fprintf('   Extracting %d species + %d compartments from %d simulations...\n', ...
    n_species - n_compartments, n_compartments, n_sims);

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

        % Extract each species/compartment
        for j = 1:n_species
            state_name = species_names{j};
            try
                [~, data, ~] = selectbyname(simdata, state_name);
                if ~isempty(data)
                    % Data found - store it
                    species_arrays{i, j} = data;
                elseif isKey(compartment_capacities, state_name)
                    % Compartment not in simdata - use constant Capacity from model
                    capacity = compartment_capacities(state_name);
                    species_arrays{i, j} = repmat(capacity, size(simdata.Time));
                else
                    % State returned empty data - store empty array
                    species_arrays{i, j} = [];
                end
            catch
                % selectbyname threw an error
                if isKey(compartment_capacities, state_name)
                    capacity = compartment_capacities(state_name);
                    species_arrays{i, j} = repmat(capacity, size(simdata.Time));
                else
                    species_arrays{i, j} = [];
                end
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
