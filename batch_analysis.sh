#!/bin/bash

# Here we retrieve the main parameters for the job 

INPUTDIR=${1}  # Directory where the input root files are
GTAG=${2}      # Global tag
NTOT=${3}      # N per file
NPC=${4}       # N per chunck
SEL=${5}       # Injections list

###########################################################
###########################################################
# You are not supposed to touch the rest of the script !!!!
###########################################################
###########################################################


# The SE directory containing the input *.p files
INDIR_XROOT=${INPUTDIR}/generators


# The SE directory containing the output EDM file with the PR output
OUTDIR=${INPUTDIR}/output/${GTAG}
mkdir $OUTDIR

echo 'The data will be read from directory: '$INDIR_XROOT
echo 'The output files will be written in: '$OUTDIR


# We loop over the data directory in order to find all the files to process

ninput=0	 
ntot=$NTOT
npc=$NPC
nsj=0

echo "#!/bin/sh" > global_stuff.sh

for ll in `ls $INDIR_XROOT | grep summary-$GTAG`
do   
    l=`basename $ll`
    echo $l
    val=0

    if [ $val = 0 ]; then
	nsj=$(( $nsj + 1))

    while [ $val -lt $ntot ]
    do
        #echo "$val"
        start=$val
        val=$(( $val + $npc ))

        stop=$val

        echo "source run_${GTAG}_${nsj}_${val}.sh"  >> global_stuff_${GTAG}.sh
	    echo "#!/bin/sh" > run_${GTAG}_${nsj}_${val}.sh
        echo "conda activate tmpprod" >> run_${GTAG}_${nsj}_${val}.sh 
	    echo "python3.9 plot_temp_loop_precess.py ${start} ${stop} ${INPUTDIR}/generators/$l $SEL $GTAG > $GTAG_${nsj}_${val}.txt" >> run_${GTAG}_${nsj}_${val}.sh
        chmod 755 run_${GTAG}_${nsj}_${val}.sh
    done
    fi
done

chmod 755 global_stuff_${GTAG}.sh

