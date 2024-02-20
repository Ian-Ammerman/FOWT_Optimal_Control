function BB = read_simoresults(forfilebin,Nt)
BB = 0;
iexistbin  = exist([forfilebin]);
  if (iexistbin)
    fid      = fopen([forfilebin],'rb');
    AA       = fread(fid,'single');
    fclose(fid);
    
    BB = reshape(AA,Nt,[]);
    disp([' ... Binary file read: ' forfilebin ])
  end