#!/bin/bash

# Bank production script for SLURM farm run at CCIN2P3

# Here we retrieve the main parameters for the job 

BKNAME=${1}  # Address of the bank file name
PFNAME=${2}  # Adress of the param file
GTAG=${3}    # Global tag
NTOT=${4}    # N per file

INPUTDIR=$PWD

# The SE directory containing the input *.p files
INDIR_XROOT=${INPUTDIR}/generators


# The SE directory containing the output EDM file with the PR output
OUTDIR=${INPUTDIR}/output/${GTAG}
mkdir $OUTDIR

ntemp=`wc -l < ${BKNAME}`

#echo $ntemp

nblocks=$(( $ntemp / $NTOT ))

#echo $nblocks

echo 'The bank info will be put in directory: '$INDIR_XROOT

echo "#!/bin/sh" > global_stuff.sh

val=0

while [ $val -le $ntemp ]
do
    echo "$val"
    start=$val
    val=$(( $val + $NTOT ))

    echo "sbatch bk_${GTAG}_${start}_${val}.sh"  >> global_stuff.sh
	echo "#!/bin/sh" > bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --job-name=test" >> bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --licenses=sps" >> bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --ntasks=1" >> bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --cpus-per-task=1" >> bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --mem-per-cpu=15GB" >> bk_${GTAG}_${start}_${val}.sh
	echo "#SBATCH --time=0-08:00" >> bk_${GTAG}_${start}_${val}.sh
    echo "source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh" >> bk_${GTAG}_${start}_${val}.sh
    echo "conda activate igwn-py39" >> bk_${GTAG}_${start}_${val}.sh
	echo "python3.9 gendata.py bank -s ${GTAG} -n ${NTOT} -start ${start} -pf $PFNAME" >> bk_${GTAG}_${start}_${val}.sh
    chmod 755 bk_${GTAG}_${start}_${val}.sh
done

chmod 755 global_stuff.sh

