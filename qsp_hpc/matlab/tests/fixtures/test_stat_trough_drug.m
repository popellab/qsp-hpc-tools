function value = test_stat_trough_drug(time, V_C_Drug)
%TEST_STAT_TROUGH_DRUG Compute trough (final) drug concentration
%
% Simple test statistic function for integration testing.

value = V_C_Drug(end);
end
