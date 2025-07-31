% This function plots slices
% and points, if provided
function plot_slices(obj, pnts)
 N = length(obj.slices);
 M = 4;
 te = 0;
 for t = 1 : N
    slice = obj.slices(t);
    
    subplot(ceil(N/M), M, t);
    plot(slice);
    title(sprintf('T=%d constraints', t));    
    xlabel(sprintf('Total Energy at T=%d', t-1));
    ylabel(sprintf('Energy at T=%d', t));    
    
    % Plot points, if provided or schedule is optimizes
    schedule = [];
    if nargin>1
        schedule = pnts;
    end
    
%    if ~any(isnan(obj.y.value))
%        schedule = obj.y.value;
%    end
    
    if ~isempty(schedule)
        hold on;        
        % Plot a dep. en line
        [e_min, e_max] = obj.getEnergyMinMax(slice, te);
        plot([te; te],[e_min, e_max],'b','LineWidth',2);
        % Plot a point
        %scatter(te, schedule(t));        
        % Update total energy
        te = te + schedule(t);        
        hold off;
    end    
 end 
end