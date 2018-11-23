function logli = GMMClassExe(par, PCAs)
% 
%     gmPDF = @(x) pdf(par.gm,x);
%     
    [~,logli] = posterior(par.gm, PCAs);

end