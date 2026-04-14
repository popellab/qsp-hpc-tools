function build_deployed()
%BUILD_DEPLOYED  Build accelerated SimFunction + mcc-package it in one session.
%
% Must run in a single MATLAB session: sf.DependentFiles points into
% ~/.MathWorks/SimBiology/<uuid>/ which MATLAB cleans up on exit, so mcc
% has to happen before this session ends.
%
% Run with: matlab -batch "build_deployed"

t_total = tic;
fprintf('[build] start\n');

% Trivial SimBiology model — exponential decay.
m = sbiomodel('poc');
c = addcompartment(m, 'cell');
addspecies(c, 'A', 1.0);
addparameter(m, 'k', 0.1);
addreaction(m, 'A -> null', 'ReactionRate', 'k*A');
cfg = getconfigset(m);
set(cfg, 'StopTime', 100);
set(cfg.SolverOptions, 'AbsoluteTolerance', 1e-9, 'RelativeTolerance', 1e-6);
fprintf('[build] model built: %.3fs\n', toc(t_total));

% SimFunction: vary 'k', observe 'A', no doses.
t = tic;
sf = createSimFunction(m, {'k'}, {'A'}, []);
fprintf('[build] SimFunction created: %.3fs\n', toc(t));

% Accelerate (codegen + MEX) — moves per-task sbioaccelerate cost offline.
t = tic;
accelerate(sf);
fprintf('[build] accelerate: %.3fs\n', toc(t));

% Save the SimFunction artifact.
if ~exist('build', 'dir'); mkdir build; end
save('build/sf_poc.mat', 'sf');

% Bundle the SimFunction artifact + every DependentFile into the CTF.
% DependentFiles is session-scoped; mcc MUST run here before session exit.
deps = sf.DependentFiles(:);
fprintf('[build] bundling %d SimFunction dependent files\n', numel(deps));

mcc_args = {'-m', 'qsp_poc_main.m', '-d', 'build', '-v', ...
            '-R', '-nodisplay,-nosplash', ...
            '-a', 'build/sf_poc.mat'};
for i = 1:numel(deps)
    mcc_args{end+1} = '-a';    %#ok<AGROW>
    mcc_args{end+1} = deps{i}; %#ok<AGROW>
end

t = tic;
mcc(mcc_args{:});
fprintf('[build] mcc: %.3fs\n', toc(t));

fprintf('[build] DONE total=%.3fs\n', toc(t_total));
end
