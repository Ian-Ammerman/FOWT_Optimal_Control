function extractNLinear(root_name,model_name,N)

% 
% Written By: Ian Ammerman
% Written: 11/3/23
%
% extractNLinear reads the OpenFAST .lin files, averages the dynamics matrices, and extracts the 
% final state space matrices to save in separate .mat files. 
%
% Inputs:
% -----------
% root_name - 
% N - number of linearization files

% ---------------------------------------------------------------------------- %

cd('Linear_Files');
% Exctract system matrices
for i = 1:N
    lin_file = sprintf('%s.%i.lin',root_name,i);
    SS_data = ReadFASTLinear(lin_file);

    A(:,:,i) = SS_data.A;

    if i == 1
        B = SS_data.B;
        C = SS_data.C;
        D = SS_data.D;
    end
end

% Average system matrices
Amid = zeros(size(A,1,2));
for j = 1:N
    Amid = Amid + A(:,:,j);
end
Afinal = Amid./N;
A = Afinal;

% Save matrices to .mat files
if contains(root_name,'HD')
    save(sprintf('%s_HD_A.mat',model_name),'A');
    save(sprintf('%s_HD_B.mat',model_name),'B');
    save(sprintf('%s_HD_C.mat',model_name),'C');
    save(sprintf('%s_HD_D.mat',model_name),'D');
else
    save(sprintf('%s_A.mat',model_name),'A');
    save(sprintf('%s_B.mat',model_name),'B');
    save(sprintf('%s_C.mat',model_name),'C');
    save(sprintf('%s_D.mat',model_name),'D');
end


save(sprintf('%s_ss_data.mat',model_name),'SS_data');
cd('..');
end