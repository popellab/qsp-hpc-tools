function value = test_stat_auc_drug(time, V_C_Drug)
%TEST_STAT_AUC_DRUG Compute AUC of drug concentration
%
% Simple test statistic function for integration testing.

value = trapz(time, V_C_Drug);
end
