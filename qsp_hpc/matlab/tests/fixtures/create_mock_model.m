function model = create_mock_model()
%CREATE_MOCK_MODEL Create a minimal SimBiology model for testing
%
% Creates a simple two-compartment PK model with:
%   - Compartments: Central (V_C), Peripheral (V_P)
%   - Species: Drug in each compartment
%   - Parameters: k_elimination, k_12, k_21
%
% This model is intentionally simple to allow fast simulation
% and predictable outputs for testing.
%
% Output:
%   model - SimBiology model object

% Create model
model = sbiomodel('TestPKModel');

% Add compartments
central = addcompartment(model, 'V_C', 50);  % 50 L central volume (constant)
peripheral = addcompartment(model, 'V_P', 100);  % 100 L peripheral volume (constant)

% Add a non-constant compartment to test extraction from simdata
% This mimics tumor volume (V_T) which can change during simulation
tumor = addcompartment(model, 'V_T', 1);  % 1 L initial tumor volume
tumor.ConstantCapacity = false;  % Mark as non-constant

% Add species to compartments
drug_central = addspecies(central, 'Drug', 0);  % Drug in central
drug_peripheral = addspecies(peripheral, 'Drug', 0);  % Drug in peripheral

% Add parameters
k_el = addparameter(model, 'k_elimination', 0.1);  % Elimination rate
k_12 = addparameter(model, 'k_12', 0.05);  % Central to peripheral
k_21 = addparameter(model, 'k_21', 0.03);  % Peripheral to central

% Add reactions
% Elimination from central compartment
r1 = addreaction(model, 'V_C.Drug -> null');
r1.ReactionRate = 'k_elimination * V_C.Drug';
r1.Name = 'Elimination';

% Distribution: Central to Peripheral
r2 = addreaction(model, 'V_C.Drug -> V_P.Drug');
r2.ReactionRate = 'k_12 * V_C.Drug';
r2.Name = 'CentralToPeripheral';

% Distribution: Peripheral to Central
r3 = addreaction(model, 'V_P.Drug -> V_C.Drug');
r3.ReactionRate = 'k_21 * V_P.Drug';
r3.Name = 'PeripheralToCentral';

% Set initial drug amount in central compartment
drug_central.InitialAmount = 100;  % 100 units initial dose

% Add a rule to make tumor volume change over time (exponential growth)
% This ensures V_T is logged in simdata as a non-constant compartment
tumor_growth = addrule(model, 'V_T = 1 * exp(0.01 * time)', 'repeatedAssignment');
tumor_growth.Name = 'TumorGrowth';

end
