function results = run_all_tests(varargin)
%RUN_ALL_TESTS Run all MATLAB unit tests for qsp-hpc-tools
%
% Usage:
%   results = run_all_tests()           % Run all tests
%   results = run_all_tests('Verbosity', 'Detailed')  % Detailed output
%   results = run_all_tests('ReportCoverage', true)   % With coverage
%
% From command line (for CI/CD):
%   matlab -batch "cd('qsp_hpc/matlab/tests'); exit(run_all_tests())"
%
% Returns:
%   results - TestResult array, or exit code (0=pass, 1=fail) in batch mode

import matlab.unittest.TestSuite;
import matlab.unittest.TestRunner;
import matlab.unittest.plugins.TAPPlugin;
import matlab.unittest.plugins.ToFile;

% Parse inputs
p = inputParser;
addParameter(p, 'Verbosity', 'Normal', @ischar);
addParameter(p, 'ReportCoverage', false, @islogical);
addParameter(p, 'TAPFile', '', @ischar);
addParameter(p, 'JUnitFile', '', @ischar);
parse(p, varargin{:});

% Get test directory
testDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(testDir);

% Add matlab directory to path for function access
addpath(matlabDir);
addpath(fullfile(testDir, 'fixtures'));

fprintf('=== QSP-HPC-Tools MATLAB Test Suite ===\n');
fprintf('Test directory: %s\n', testDir);
fprintf('MATLAB directory: %s\n\n', matlabDir);

% Create test suite from test directory
suite = TestSuite.fromFolder(testDir, 'IncludingSubfolders', false);

fprintf('Found %d test classes\n\n', length(suite));

% Create test runner with appropriate verbosity
switch lower(p.Results.Verbosity)
    case 'detailed'
        runner = TestRunner.withTextOutput('Verbosity', 3);
    case 'minimal'
        runner = TestRunner.withTextOutput('Verbosity', 1);
    otherwise
        runner = TestRunner.withTextOutput('Verbosity', 2);
end

% Add TAP plugin for CI/CD integration if requested
if ~isempty(p.Results.TAPFile)
    tapFile = p.Results.TAPFile;
    runner.addPlugin(TAPPlugin.producingOriginalFormat(ToFile(tapFile)));
    fprintf('TAP output: %s\n', tapFile);
end

% Add JUnit XML plugin if requested
if ~isempty(p.Results.JUnitFile)
    import matlab.unittest.plugins.XMLPlugin;
    junitFile = p.Results.JUnitFile;
    runner.addPlugin(XMLPlugin.producingJUnitFormat(junitFile));
    fprintf('JUnit XML output: %s\n', junitFile);
end

% Run tests
results = runner.run(suite);

% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Total:  %d\n', length(results));
fprintf('Passed: %d\n', sum([results.Passed]));
fprintf('Failed: %d\n', sum([results.Failed]));
fprintf('Incomplete: %d\n', sum([results.Incomplete]));

% Calculate duration
totalDuration = sum([results.Duration]);
fprintf('Duration: %.2f seconds\n', totalDuration);

% Return exit code in batch mode
if batchStartupOptionUsed
    if sum([results.Failed]) > 0 || sum([results.Incomplete]) > 0
        results = 1;  % Failure
    else
        results = 0;  % Success
    end
end

end
