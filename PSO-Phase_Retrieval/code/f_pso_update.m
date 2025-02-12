function pso_state = f_pso_update(pso_state,iter)

% Retrieve relevant PSO parameters and state
particles = pso_state.particles;
CostFunction = pso_state.CostFunction;
nVar = pso_state.nVar;
VarMin = pso_state.VarMin;
VarMax = pso_state.VarMax;
gbest = pso_state.gbest;
fprintf('Current gbest cost: %.6f\n', gbest.cost);
w = pso_state.w;
c1 = pso_state.c1;
c2 = pso_state.c2;

for i = 1:numel(particles)
    % Update velocity
    particles(i).velocity = w * particles(i).velocity ...
        + c1 * rand(1,nVar) .* (particles(i).best.position - particles(i).position) ...
        + c2 * rand(1,nVar) .* (gbest.position - particles(i).position);

    % Update position
    particles(i).position = particles(i).position + particles(i).velocity;

    % Apply bounds
    particles(i).position = max(particles(i).position, VarMin);
    particles(i).position = min(particles(i).position, VarMax);

    % Re-evaluate cost
    particles(i).cost = CostFunction(particles(i).position);

    % Update personal best
    if particles(i).cost < particles(i).best.cost
        particles(i).best.position = particles(i).position;
        particles(i).best.cost = particles(i).cost;

        % Update global best
        if particles(i).best.cost < gbest.cost
            gbest = particles(i).best;
        end
    end
    % fprintf('Iter: %d, Particle %d, Position: [%f, %f], Cost: %.6f\n', ...
    % iter, i, particles(i).position(1), particles(i).position(2), particles(i).cost);
end
fprintf('Best position: [%f, %f]',gbest.position(1),gbest.position(2))
% Update PSO state
pso_state.particles = particles;
pso_state.gbest = gbest;
end
