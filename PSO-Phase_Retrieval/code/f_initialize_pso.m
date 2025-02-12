function pso_state = f_initialize_pso(psoParams, CostFunction)

% Particle template
x0.position = [];
x0.velocity = [];
x0.cost = [];
x0.best.position = [];
x0.best.cost = [];

% Create particle array
particle = repmat(x0, psoParams.nPop, 1);

% Initialize global best
gbest.cost = inf;

% Initialize particles
for i = 1:psoParams.nPop
    particle(i).position = psoParams.VarMin + (psoParams.VarMax - psoParams.VarMin) .* rand(1, psoParams.nVar);
    particle(i).velocity = (psoParams.VarMax - psoParams.VarMin) .* (rand(1, psoParams.nVar) - 0.5);
    particle(i).cost = CostFunction(particle(i).position);
    particle(i).best = particle(i);

    fprintf('Particle %d: Initial cost = %.6f, Initial position = [%f, %f]\n', i, ...
    particle(i).cost, particle(i).position(1), particle(i).position(2));

    % Update global best
    if particle(i).cost < gbest.cost
        gbest = particle(i).best;
    end
end

% Initialize PSO state
pso_state.particles = particle;
pso_state.gbest = gbest;
pso_state.w = psoParams.w;
pso_state.c1 = psoParams.c1;
pso_state.c2 = psoParams.c2;
pso_state.CostFunction = CostFunction;
pso_state.nVar = psoParams.nVar;
pso_state.VarMin = psoParams.VarMin;
pso_state.VarMax = psoParams.VarMax;

end
