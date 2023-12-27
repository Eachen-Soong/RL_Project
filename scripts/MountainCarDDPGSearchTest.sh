taus=(0.001 0.005 0.01 0.1 1)
exploration_noises=(0.1 0.25 0.5 0.75 1.0)

for tau in "${taus[@]}"
do
    for noise in "${exploration_noises[@]}"
    do
        echo "Running with tau=$tau and exploration_noise=$noise"
        python MountainCar/DDPG.py --mode test --max_episode 300 --tau $tau --exploration_noise $noise --seed True
    done
done