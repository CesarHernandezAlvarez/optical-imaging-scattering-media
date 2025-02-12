function psoParams = f_config_pso(pso_iter_no,part_no)
    psoParams.MaxIt = 200;
    psoParams.nPop = part_no;
    fprintf("PSO Iterations: %d, Population: %d\n",pso_iter_no,psoParams.nPop);
    psoParams.w = 0.9;
    psoParams.damp = 0.99;
    psoParams.c1 = 2;
    psoParams.c2 = 2;
    psoParams.ShowIterInfo = true;
end