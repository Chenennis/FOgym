function agg_model = aggregate_determ_final(obj, state, params)

fprintf('Aggregation.');

grpDFOs = DetDFOSystem.empty(length(state.groups), 0);

% Loop throught the resulting groups, apply unification and aggregation
cnt = 0; tm = tic;
for i = 1 : length(state.groups)
    
 % Running aggregate 
 a_slices = state.groups{i}(1).slices;
 
 % Sort groups - no effect
 % sim = arrayfun(@(s) obj.aggregate_slices_determ3(a_slices, s.slices), state.groups{i}(2:end));
 % [sim, idx] =  sort(sim, 'descend'); % , 
 % state.groups{i}(2:end) = state.groups{i}(idx);
   
 f_cnt = length(state.groups{i});

 cnt = cnt + 1;
 for j = 2 : f_cnt
     % Get slices of the incrementally aggregated DFO
     slices = state.groups{i}(j).slices;
     
     % Unify slices if needed
     if state.unify_slices
       % a_slices
       slices  = obj.unify_slices(state.groups{i}(1).slices, slices);
       if isempty(slices)
           error('Failed unifying slices');            % Failed to unify slices
       end
     end        
    
     % Perform aggregation
     [~, ~, a_slices] = obj.aggregate_slices_determ3(a_slices, slices, state.sample_count);     
    
     % Show status
%      cnt = cnt + 1;
%      if mod(cnt, state.num_models / 10) == 0 || toc(tm) >= 5
%             fprintf('%2.2f%% ', cnt / state.num_models * 100);
%             tm = tic;
%      end
 end;
 
 % Apply normalization and compaction of slices
 p_slices = [];
 for k = 1 : length(a_slices)
     p_slices = [p_slices; Polyhedron('V', a_slices(k).V).minVRep().minHRep().normalize()];
 end
 
 % Build the aggregated DFO
 grpDFOs(i) = DetDFOSystem(p_slices, state.groups{i});

end;

if length(grpDFOs) == 1
    agg_model = grpDFOs(1);
else
    agg_model = MultiSystem(grpDFOs);
end

%  
%  if state.unify_slices
%      % Build an outer approximation
%      OA = state.groups{i}(1);
%      for j = 2 : f_cnt
%          OA = DFOSystem(MultiSystem(OA, state.groups{i}(j)), 0);
%      end
%      
%      a_slices  = obj.unify_slices(OA.slices, a_slices);
%  end
end