#!/bin/bash
#SBATCH -J conv_autoenc           # job name
#SBATCH -o conv_autoenc.o       # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of mpi tasks requested
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=manan.doshi96@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

python main.py -n5 -l0.0004 -e10000 --roughness 2 --area 500 -r0.4 --length 40 -b 32 -l4e-4 --restart --log
python main.py -n5 -l0.0004 -e5000  --roughness 2 --area 500 -r0.4 --length 40 -b 32 -l4e-4 --log

python main.py -n6 -l0.0004 -e10000 --roughness 2 --area 500 -r0.4 --length 40 -b 128 -l4e-4 --restart --log
python main.py -n6 -l0.0004 -e5000  --roughness 2 --area 500 -r0.4 --length 40 -b 128 -l4e-4 --log

python main.py -n7 -l0.0004 -e10000 --roughness 2 --area 500 -r0.4 --length 40 -b 256 -l4e-4 --restart --log
python main.py -n7 -l0.0004 -e5000  --roughness 2 --area 500 -r0.4 --length 40 -b 256 -l4e-4 --log

