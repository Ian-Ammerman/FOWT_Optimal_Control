function extractLinear(file_name,root_name)

% 
% Written By: Ian Ammerman
% Written: 9/1/23
%
% extractLinear reads the OpenFAST .lin file and extracts the state space
% matrices to save in separate .mat files.
%
% Inputs:
% -----------
% file_name - name of .lin file

% ---------------------------------------------------------------------------- %

lin_file = file_name;
SS_data = ReadFASTLinear(lin_file);

A = SS_data.A;
B = SS_data.B;
C = SS_data.C;
D = SS_data.D;

if contains(file_name,'HD')
    save(sprintf('%s_HD_A.mat',root_name),'A');
    save(sprintf('%s_HD_B.mat',root_name),'B');
    save(sprintf('%s_HD_C.mat',root_name),'C');
    save(sprintf('%s_HD_D.mat',root_name),'D');
else
    save(sprintf('%s_A.mat',root_name),'A');
    save(sprintf('%s_B.mat',root_name),'B');
    save(sprintf('%s_C.mat',root_name),'C');
    save(sprintf('%s_D.mat',root_name),'D');
end


save(sprintf('%s_ss_data.mat',root_name),'SS_data');

end