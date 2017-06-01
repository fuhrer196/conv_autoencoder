#!/bin/bash
#SBATCH -J conv_autoenc3           # job name
#SBATCH -o conv_autoenc3.o       # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of mpi tasks requested
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=manan.doshi96@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

python main.py -n11 -l0.0004 -e10000 --roughness 2 --area 500 -r0.4 --length 40 -b 128 -l4e-5 --restart --log
python main.py -n11 -l0.0004 -e5000  --roughness 2 --area 500 -r0.4 --length 40 -b 128 -l4e-5 --log

python main.py -n12 -l0.0004 -e10000 --roughness 2 --area 500 -r0.6 --length 40 -b 128 -l4e-5 --restart --log
python main.py -n12 -l0.0004 -e5000  --roughness 2 --area 500 -r0.6 --length 40 -b 128 -l4e-5 --log

python main.py -n13 -l0.0004 -e10000 --roughness 2 --area 500 -r0.8 --length 40 -b 128 -l4e-4 --restart --log
python main.py -n13 -l0.0004 -e5000  --roughness 2 --area 500 -r0.8 --length 40 -b 128 -l4e-4 --log
