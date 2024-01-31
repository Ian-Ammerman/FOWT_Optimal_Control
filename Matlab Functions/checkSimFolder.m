function checkSimFolder(sim_folder,root_name)

%
% Written By: Ian Ammerman
% Written: 9/1/23
%
% checkSimFolder stops the simulation if an old output is detected in the
% target simulation directory.

start_dir = pwd;
cd(sim_folder)

out_name = sprintf('%s.out',root_name);
if isfile(out_name)
    errordlg("Old output detected in simulation directory.",'Simulation Folder Error');
    return
end

cd(start_dir)
end