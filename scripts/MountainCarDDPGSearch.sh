taus=(0.001 0.005 0.01)
exploration_noises=(0.1 0.2 0.3)

for tau in "${taus[@]}"
do
    for noise in "${exploration_noises[@]}"
    do
        echo "Running with tau=$tau and exploration_noise=$noise"
        python MountainCar/DDPG.py --tau $tau --exploration_noise $noise
    done
done