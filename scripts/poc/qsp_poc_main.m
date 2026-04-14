function qsp_poc_main(n_sims_str, sf_path)
%QSP_POC_MAIN  Compiled deployment-overhead probe using a SimFunction.
%
% Built with: mcc -m qsp_poc_main.m -a build/sf_poc.mat
% Run with:   ./qsp_poc_main 100 build/sf_poc.mat
%
% Loads a pre-built SimFunction artifact and invokes it n_sims times with
% varying parameter values. Times each phase so we can compare the
% deployed-exe floor against batch_worker.m's [timing-summary] floor.

t_total = tic;

if nargin < 1 || isempty(n_sims_str); n_sims = 1; else; n_sims = str2double(n_sims_str); end
if nargin < 2 || isempty(sf_path);    sf_path = 'build/sf_poc.mat'; end

fprintf('[POC] entry  : %.3fs (n_sims=%d, sf=%s)\n', toc(t_total), n_sims, sf_path);

t = tic;
S = load(sf_path, 'sf');
sf = S.sf;
fprintf('[POC] load   : %.3fs\n', toc(t));

% First sim (cold) — picks up any deferred MEX/JIT cost on this process.
t = tic;
[~, ~] = sf(0.1, 100);
fprintf('[POC] sim 1  : %.3fs\n', toc(t));

% Subsequent sims (warm) with varying k.
warm_total = 0;
ks = linspace(0.05, 0.5, max(n_sims - 1, 1));
for i = 2:n_sims
    t = tic;
    [~, ~] = sf(ks(i - 1), 100);
    warm_total = warm_total + toc(t);
end
if n_sims > 1
    fprintf('[POC] sim 2-%d: %.3fs (avg %.3fs/sim)\n', n_sims, warm_total, warm_total/(n_sims-1));
end

fprintf('[POC-summary] total=%.3fs n_sims=%d\n', toc(t_total), n_sims);
end
