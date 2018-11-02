function hiddenlayers = hl_try(hlnum)

    if hlnum <= 3
        hiddenlayers = [hlnum];
    elseif hlnum > 3 && hlnum <= 8
        lastl = min(round(hlnum/2),3);
        hiddenlayers = [hlnum - lastl,lastl];
    elseif hlnum > 8
        lastl = 3;
        hlleft = hlnum - lastl;
        hlmid = min(round(hlleft/2), 5);
        hiddenlayers = [hlleft - hlmid, hlmid, lastl];
    end
            
        
end
