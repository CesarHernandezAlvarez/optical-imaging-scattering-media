function out = f_pso_min(problem, psoParams)

%% Problem definition
CostFunction = problem.CostFunction;

%% Parameters of PSO
MaxIt = psoParams.MaxIt;                   
update_freq = psoParams.update_freq;

%% Main loop of PSO
BestCosts = zeros(MaxIt, 1);

for iter = 1:MaxIt
    % Update PSO state every `update_freq` iterations
    if mod(iter, update_freq) == 0
        % Damping inertia coefficient only when the system is updated
        psoParams.pso_state.w = psoParams.pso_state.w * psoParams.damp;
        psoParams.pso_state = f_pso_update(psoParams.pso_state,psoParams.iter);
    end

    % Evaluate and track best cost
    BestCosts(iter) = psoParams.pso_state.gbest.cost;
    % fprintf('Iteration %d: Best Cost: %.6f, sigmaGF: %.6f, threshold: %.6f\n', ...
    %     iter, BestCosts(iter), psoParams.pso_state.gbest.position(1), psoParams.pso_state.gbest.position(2));

    % Display iteration information
    if psoParams.ShowIterInfo && mod(iter, update_freq) == 0
        fprintf('Iteration: %d, Best Cost: %.6f\n', iter/update_freq, BestCosts(iter));
    end

    psoParams.iter = psoParams.iter + 1;
end

% Output results
out.pop = psoParams.pso_state.particles;
out.BestSol = psoParams.pso_state.gbest;
out.BestCosts = BestCosts;

end