#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=2:00:00
#SBATCH --job-name asteroids_serialx40
#SBATCH --output=asteroids_1_40_%j.txt
#SBATCH --mail-type=FAIL

module load python/3.8.5
source '/home/r/rbond/jorlo/actmadcows/bin/activate'
cd $SLURM_SUBMIT_DIR

python3.8 ~/dev/minorplanets/asteroid_movie_ricco.py /gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/Iris.npy iris "/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/*/*_map.fits" /scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/iris







cd $SLURM_SUBMIT_DIR

iter=$(($SLURM_ARRAY_TASK_ID*40))
# EXECUTION COMMAND; ampersand off 20 jobs and wait
task=$(expr $iter + 0) && name=$(sed -n "${task}p" asteroid_list.txt) && python3.8 ~/dev/minorplanets/asteroid_movie.py /gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/$name.npy iris "/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/*/*_map.fits" /scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/$name &
task=$(expr $iter + 1) && python3 run_stack.py $task &
task=$(expr $iter + 2) && python3 run_stack.py $task &
task=$(expr $iter + 3) && python3 run_stack.py $task &
task=$(expr $iter + 4) && python3 run_stack.py $task &
task=$(expr $iter + 5) && python3 run_stack.py $task &
task=$(expr $iter + 6) && python3 run_stack.py $task &
task=$(expr $iter + 7) && python3 run_stack.py $task &
task=$(expr $iter + 8) && python3 run_stack.py $task &
task=$(expr $iter + 9) && python3 run_stack.py $task &
task=$(expr $iter + 10) && python3 run_stack.py $task &
task=$(expr $iter + 11) && python3 run_stack.py $task &
task=$(expr $iter + 12) && python3 run_stack.py $task &
task=$(expr $iter + 13) && python3 run_stack.py $task &
task=$(expr $iter + 14) && python3 run_stack.py $task &
task=$(expr $iter + 15) && python3 run_stack.py $task &
task=$(expr $iter + 16) && python3 run_stack.py $task &
task=$(expr $iter + 17) && python3 run_stack.py $task &
task=$(expr $iter + 18) && python3 run_stack.py $task &
task=$(expr $iter + 19) && python3 run_stack.py $task &
task=$(expr $iter + 20) && python3 run_stack.py $task &
task=$(expr $iter + 21) && python3 run_stack.py $task &
task=$(expr $iter + 22) && python3 run_stack.py $task &
task=$(expr $iter + 23) && python3 run_stack.py $task &
task=$(expr $iter + 24) && python3 run_stack.py $task &
task=$(expr $iter + 25) && python3 run_stack.py $task &
task=$(expr $iter + 26) && python3 run_stack.py $task &
task=$(expr $iter + 27) && python3 run_stack.py $task &
task=$(expr $iter + 28) && python3 run_stack.py $task &
task=$(expr $iter + 29) && python3 run_stack.py $task &
task=$(expr $iter + 30) && python3 run_stack.py $task &
task=$(expr $iter + 31) && python3 run_stack.py $task &
task=$(expr $iter + 32) && python3 run_stack.py $task &
task=$(expr $iter + 33) && python3 run_stack.py $task &
task=$(expr $iter + 34) && python3 run_stack.py $task &
task=$(expr $iter + 35) && python3 run_stack.py $task &
task=$(expr $iter + 36) && python3 run_stack.py $task &
task=$(expr $iter + 37) && python3 run_stack.py $task &
task=$(expr $iter + 38) && python3 run_stack.py $task &
task=$(expr $iter + 39) && python3 run_stack.py $task &

wait
