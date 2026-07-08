#!/bin/sh

export DOCKER_IMAGE=132919042661.dkr.ecr.ap-northeast-1.amazonaws.com/nvcr.io/nvidia/tensorflow:25.02-tf2-py3
export EXEC_MODE='batch'
export PARTITION=gpu-g6e

#Replace your jobscript appropriately
export JOB_SCRIPT=submit/exp1.sh

export nGPUS=1               # P4D max:8 GPUS
export nCPUS=12
export nMEM=64G               # default: MBytes
export SHM_SIZE=64G
export Walltime='336:00:00'     # HH:MM:SS
export MAIL_TO=''

##### Check JOB Settings
/ff/scsk/slurm/bin/check_jobenv
ret=$?
if [ $ret -ne  0 ]; then
        echo "ERROR: $ret"
        exit $ret
fi

sbatch -p ${PARTITION} --gpus-per-node=${nGPUS} --mem=${nMEM} -c ${nCPUS} -t ${Walltime} --mail-user=${MAIL_TO} --export ALL /ff/scsk/slurm/bin/userjob/run_batch.sh

#Replace your jobscript appropriately
export JOB_SCRIPT=submit/exp2.sh

export nGPUS=1               # P4D max:8 GPUS
export nCPUS=12
export nMEM=64G               # default: MBytes
export SHM_SIZE=64G
export Walltime='336:00:00'     # HH:MM:SS
export MAIL_TO=''

##### Check JOB Settings
/ff/scsk/slurm/bin/check_jobenv
ret=$?
if [ $ret -ne  0 ]; then
        echo "ERROR: $ret"
        exit $ret
fi

sbatch -p ${PARTITION} --gpus-per-node=${nGPUS} --mem=${nMEM} -c ${nCPUS} -t ${Walltime} --mail-user=${MAIL_TO} --export ALL /ff/scsk/slurm/bin/userjob/run_batch.sh
