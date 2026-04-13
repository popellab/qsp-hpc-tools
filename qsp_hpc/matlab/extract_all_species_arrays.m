function species_data = extract_all_species_arrays(chunk_results, model)
%EXTRACT_ALL_SPECIES_ARRAYS Extract time, species, rule params, and compartments from simulation results
%
% This function extracts full simulation outputs (time, all species,
% non-constant parameters from repeatedAssignment rules, and compartment
% volumes) from SimBiology simulation results for saving to persistent storage.
%
% Inputs:
%   chunk_results  - Struct array with .simData fields (SimData objects)
%   model          - SimBiology model object (for species names)
%
% Outputs:
%   species_data   - Struct with fields:
%                    .n_sims         - Number of simulations
%                    .n_timepoints   - Number of time points per simulation
%                    .n_species      - Number of species + rule params + compartments
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
% Separate constant compartments (use Capacity) from non-constant ones (extract from simdata)
all_compartments = sbioselect(model, 'Type', 'compartment');
compartment_names = cell(1, length(all_compartments));
constant_compartments = containers.Map();  % Maps name -> Capacity for constant compartments only
for i = 1:length(all_compartments)
    comp = all_compartments(i);
    compartment_names{i} = comp.Name;
    % Check if compartment has constant capacity (ConstantCapacity property)
    % If true, we can use the model's Capacity value directly without calling selectbyname
    if comp.ConstantCapacity
        constant_compartments(comp.Name) = comp.Capacity;
    end
end

% Also get non-constant parameters (repeatedAssignment rule targets like pO2, HIF, glc)
% These are algebraic parameters that appear in simData but are not type 'species'.
% They are created with addparameter(..., 'ConstantValue', false) + addrule(..., 'repeatedAssignment').
all_params = sbioselect(model, 'Type', 'parameter');
rule_param_names = {};
for i = 1:length(all_params)
    if ~all_params(i).ConstantValue
        rule_param_names{end+1} = all_params(i).Name; %#ok<AGROW>
    end
end

% Combine species, rule parameters, and compartment names
species_names = [species_names, rule_param_names, compartment_names];
n_species = length(species_names);
n_compartments = length(compartment_names);
n_rule_params = length(rule_param_names);

n_constant = length(constant_compartments);
n_nonconstant = n_compartments - n_constant;
fprintf('   Extracting %d species + %d rule params + %d compartments (%d constant, %d non-constant) from %d simulations...\n', ...
    n_species - n_compartments - n_rule_params, n_rule_params, n_compartments, n_constant, n_nonconstant, n_sims);
if n_rule_params > 0
    fprintf('   Rule parameters: %s\n', strjoin(rule_param_names, ', '));
end

% Initialize output arrays
time_arrays = cell(n_sims, 1);
species_arrays = cell(n_sims, n_species);
status = zeros(n_sims, 1);

% Pre-compute per-species constant-compartment flags + values for parfor safety.
% containers.Map doesn't serialize cleanly to parfor workers, and state names
% like 'V_T.C1' can't be struct fields. Flat arrays handle both issues.
is_const_comp = false(1, n_species);
const_comp_vals = zeros(1, n_species);
for j = 1:n_species
    sname = species_names{j};
    if isKey(constant_compartments, sname)
        is_const_comp(j) = true;
        const_comp_vals(j) = constant_compartments(sname);
    end
end

% Extract data from each simulation. Uses parfor when MATLAB_WORKERS > 0 and a
% parpool is already open (expected when invoked from batch_worker).
num_workers = str2double(getenv('MATLAB_WORKERS'));
if isnan(num_workers) || num_workers < 0
    num_workers = 0;
end
p = gcp('nocreate');
if isempty(p)
    % No pool active — run serially regardless of env var.
    num_workers = 0;
end

t_extract = tic;
parfor (i = 1:n_sims, num_workers)
    simdata = chunk_results(i).simData;

    if isempty(simdata)
        % Failed simulation - store empty arrays
        status(i) = -1;
        time_arrays{i} = [];
        row_cells = cell(1, n_species);
        for j = 1:n_species
            row_cells{j} = [];
        end
    else
        % Successful simulation - extract time and species
        status(i) = 1;
        time_arrays{i} = simdata.Time;

        % Extract each species/compartment into a row cell, then assign once
        % (parfor's slicing analyzer prefers a single assignment per iter).
        row_cells = cell(1, n_species);
        for j = 1:n_species
            if is_const_comp(j)
                row_cells{j} = repmat(const_comp_vals(j), size(simdata.Time));
            else
                [~, data, ~] = selectbyname(simdata, species_names{j});
                if ~isempty(data)
                    row_cells{j} = data;
                else
                    row_cells{j} = [];
                end
            end
        end
    end
    species_arrays(i, :) = row_cells;
end
fprintf('   Extraction loop: %.1fs (workers=%d)\n', toc(t_extract), num_workers);

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
