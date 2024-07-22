#!/bin/bash

# Bank production script for multi-CPU run (using eg. parallel)

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

echo 'The bank info will be put in directory: '$INDIR_XROOT

echo "#!/bin/sh" > global_prod.sh

val=0

while [ $val -le $ntemp ]
do
    echo "$val"
    start=$val
    val=$(( $val + $NTOT ))

	echo "python3.9 gendata.py bank -s ${GTAG} -n ${NTOT} -start ${start} -pf $PFNAME" >> global_stuff.sh
done

chmod 755 global_stuff.sh

