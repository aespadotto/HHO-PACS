#!/bin/bash
#
# Execute hho file on series of meshes, and calculate outputs
#

###
# Executable file for the considered scheme
executable_name="hho-magnetostatics-spadotto"

errorsfile="data_rates.dat"
csvfile="data_rates.csv"
latexfile="rates.tex"

###
# Directories
origin=$(pwd)
if [ ! -f ../directories.sh ]; then
  echo "directories.sh does not exist. Please read the README.txt in the parent folder."
  exit
fi
. ../directories.sh

# Options:
if [[ $1 == "help" ]]; then
	echo -e "\nExecute tests using parameters in data.sh, creates and compile latex file, and calculate rates.\n
Executed without parameters: uses the data in the local data.sh file\n
Parameters that can be passed: data file | -k <value of k> | -l <value of l>\n"
	exit;
fi;

outdir="output-"$(date +"%m-%d")$(date +"-%H-%M")
if [ -d $outdir ]; then
	\rm -r $outdir
fi
	mkdir $outdir

###
# LOAD DATA
# (LATER: Test that each required data exists (files, parameters...))
# Default data file
datafile=$(readlink -f data.sh);
# We go over the parameters. -k K or -l L will set k or l to these values. Any other
# parameter is assumed to be a datafile that overrides the previous one
until [ $# -eq 0 ]; do
  if [[ $1 == "-k" ]]; then
    ksave=$(echo $2 | sed s/'^[^0-9].*'//g)
    shift 2
  elif [[ $1 == "-l" ]]; then
    lsave=$(echo $2 | sed s/'^[^0-9].*'//g)
    shift 2
  else
    datafile=$(readlink -f $1)
    shift
  fi
done
echo -e "Using data file $datafile\n"
. $datafile

# if ksave or lsave have been ecountered we override whatever values $datafile might have contained
if [[ ! -z "$ksave" ]]; then
  k=$ksave;
fi
if [[ ! -z "$lsave" ]]; then
  l=$lsave;
fi



k=2
###
# EXECUTE SEQUENCE FOR EACH DEGREE
for kk in `seq 0 $k`;
do
echo "degrees: (face) k=$kk"
###
# EXECUTE FOR EACH MESH
nbmesh=${#mesh[@]}
for i in `seq 1 $nbmesh`;
do
	meshtype=$(echo ${mesh[$i]} | cut -d ':' -f 1)
	meshfile=$meshdir"/"$(echo ${mesh[$i]} | cut -d ':' -f 2)
	echo -e "\n*************************\nmesh $i out of $nbmesh: ${mesh[$i]}"
	# Execute code
	$executable -t $meshtype -m $meshfile -k $kk -s -bc Mixed
  # Move outputs
  mv results.txt $outdir/results-$i-$kk.txt
done
#for mesh
done
#for degree

# PREPARE DATA FOR LATEX DOCUMENT
cd $outdir


for kk in `seq 0 $k`; do
>$errorsfile

printf "MeshSize Reasidual EnergyError L2Error MeshReg NoDofs " >> $errorsfile
printf "\n">>$errorsfile
for i in `seq 1 $nbmesh`; do
  MeshSize=$(awk '/MeshSize:/ {print $NF}' results-$i-$kk.txt)
  Residual=$(awk '/Residual:/ {print $NF}' results-$i-$kk.txt)
  EnergyError=$(awk '/EnergyError:/ {print $NF}' results-$i-$kk.txt)
  L2Error=$(awk '/L2Error:/ {print $NF}' results-$i-$kk.txt)
	MeshReg=$(awk '/MeshReg:/ {print $NF}' results-$i-$kk.txt)
  NoDofs=$(awk '/NoDofs:/ {print $NF}' results-$i-$kk.txt)
	printf  "$MeshSize $Residual $EnergyError $L2Error $MeshReg  $NoDofs " >> $errorsfile
  printf "\n">>$errorsfile
done
cp $errorsfile data_rates-$kk.dat
done
## CREATE AND COMPILE LATEX

ratio=0.4


echo -e "\\\documentclass{article}

\\\usepackage{amsfonts,latexsym,graphicx, csvsimple}

\\\setlength{\\\textwidth}{16cm}
\\\setlength{\\\textheight}{23cm}
\\\setlength{\\\oddsidemargin}{0cm}
\\\setlength{\\\evensidemargin}{0cm}
\\\setlength{\\\topmargin}{-1cm}
\\\parindent=0pt

\\\usepackage{pgfplots,pgfplotstable}
\\\usepackage{caption, subcaption}
\\\usepackage{mwe}
\\\usepackage{adjustbox}
\\\usetikzlibrary{calc,external}


\\\pgfplotsset{every axis legend/.append style={at={(0.5,1.03)}, anchor=south}}

\\\newcommand{\\\logLogSlopeTriangle}[5]
{
    \pgfplotsextra
    {
        \\\pgfkeysgetvalue{/pgfplots/xmin}{\\\xmin}
        \\\pgfkeysgetvalue{/pgfplots/xmax}{\\\xmax}
        \\\pgfkeysgetvalue{/pgfplots/ymin}{\\\ymin}
        \\\pgfkeysgetvalue{/pgfplots/ymax}{\\\ymax}

        % Calculate auxilliary quantities, in relative sense.
        \\\pgfmathsetmacro{\\\xArel}{#1}
        \\\pgfmathsetmacro{\\\yArel}{#3}
        \\\pgfmathsetmacro{\\\xBrel}{#1-#2}
        \\\pgfmathsetmacro{\\\yBrel}{\\\yArel}
        \\\pgfmathsetmacro{\\\xCrel}{\\\xArel}

        \\\pgfmathsetmacro{\\\lnxB}{\\\xmin*(1-(#1-#2))+\xmax*(#1-#2)} % in [xmin,xmax].
        \\\pgfmathsetmacro{\\\lnxA}{\\\xmin*(1-#1)+\xmax*#1} % in [xmin,xmax].
        \\\pgfmathsetmacro{\\\lnyA}{\\\ymin*(1-#3)+\ymax*#3} % in [ymin,ymax].
        \\\pgfmathsetmacro{\\\lnyC}{\\\lnyA+#4*(\\\lnxA-\\\lnxB)}
        \\\pgfmathsetmacro{\\\yCrel}{\\\lnyC-\\\ymin)/(\\\ymax-\\\ymin)}

        % Define coordinates for \draw. MIND THE 'rel axis cs' as opposed to the 'axis cs'.
        \\\coordinate (A) at (rel axis cs:\\\xArel,\\\yArel);
        \\\coordinate (B) at (rel axis cs:\\\xBrel,\\\yBrel);
        \\\coordinate (C) at (rel axis cs:\\\xCrel,\\\yCrel);

        % Draw slope triangle.
        \\\draw[#5]   (A)-- node[pos=0.5,anchor=north] {\\\scriptsize{1}}
                    (B)--
                    (C)-- node[pos=0.,anchor=west] {\\\scriptsize{#4}} %% node[pos=0.5,anchor=west] {#4}
                    cycle;
    }
}

\\\begin{document}

\\\begin{figure*}
\\\centering

   \\\begin{subfigure}[b]{0.4
   \\\textwidth}
   \\\centering
   \\\begin{adjustbox}{width=\linewidth}
    \\\begin{tikzpicture}
      \\\begin{loglogaxis}[xlabel=\$h\$, ylabel=EnergyError,  legend columns=3] ">> $latexfile

      for kk in `seq 0 $k`; do
      echo -e   "\\\addplot table[x=MeshSize, y=EnergyError] {data_rates-$kk.dat};
        \\\addlegendentry{\$k = \$ $kk}" >> $latexfile
      done

  echo -e      "\\\logLogSlopeTriangle{0.90}{0.4}{0.1}{1}{black};
        \\\logLogSlopeTriangle{0.90}{0.4}{0.1}{2}{black};
        \\\logLogSlopeTriangle{0.90}{0.4}{0.1}{3}{black};
      \\\end{loglogaxis}
\\\end{tikzpicture}
\\\end{adjustbox}
\\\end{subfigure}
\\\hfill
\\\begin{subfigure}[b]{0.4
\\\textwidth}
\\\centering
\\\begin{adjustbox}{width=\linewidth}
\\\begin{tikzpicture}
      \\\begin{loglogaxis}[xlabel=\$h\$, ylabel=L2Error, legend columns=3] ">> $latexfile

      for kk in `seq 0 $k`; do
      echo -e   "\\\addplot table[x=MeshSize, y=L2Error] {data_rates-$kk.dat};
        \\\addlegendentry{\$k = \$ $kk}" >> $latexfile
      done

  echo -e      "\\\logLogSlopeTriangle{0.90}{0.4}{0.1}{2}{black};
        \\\logLogSlopeTriangle{0.90}{0.4}{0.1}{3}{black};
        \\\logLogSlopeTriangle{0.90}{0.4}{0.1}{4}{black};
      \\\end{loglogaxis}


    \\\end{tikzpicture}
    \\\end{adjustbox}
    \\\end{subfigure}
    \\\vskip
    \\\baselineskip
    \\\begin{subfigure}[b]{0.4
    \\\textwidth}
    \\\centering
    \\\begin{adjustbox}{width=\linewidth}
    \\\begin{tikzpicture}
          \\\begin{loglogaxis}[xlabel=DOFS, ylabel=EnergyError, legend columns=3] ">> $latexfile

          for kk in `seq 0 $k`; do
          echo -e   "\\\addplot table[x=NoDofs, y=EnergyError] {data_rates-$kk.dat};
            \\\addlegendentry{\$k = \$ $kk}" >> $latexfile
          done

      echo -e      "\\\end{loglogaxis}
      \\\end{tikzpicture}
      \\\end{adjustbox}
      \\\end{subfigure}
      \\\hfill
      \\\begin{subfigure}[b]{0.4
      \\\textwidth}
      \\\centering
      \\\begin{adjustbox}{width=\linewidth}
      \\\begin{tikzpicture}
            \\\begin{loglogaxis}[xlabel=DOFS, ylabel=L2Error, legend columns=3] ">> $latexfile

            for kk in `seq 0 $k`; do
            echo -e   "\\\addplot table[x=NoDofs, y=L2Error] {data_rates-$kk.dat};
              \\\addlegendentry{\$k = \$ $kk}" >> $latexfile
            done

        echo -e      "\\\end{loglogaxis}


        \\\end{tikzpicture}
        \\\end{adjustbox}
        \\\end{subfigure}


    \\\caption{Cubic mesh sequence}



\\\end{figure*}" >> $latexfile


#create a csv file of data to make latex table
for kk in `seq 0 $k`; do
>$csvfile
cp $csvfile data_rates-$kk.csv
awk -v kkk=$kk '{for (i=1;i<NF; i++) {printf $i","}; printf $NF; printf"\n"}' data_rates-$kk.dat>>data_rates-$kk.csv
echo -e "Degrees:  \$k=$kk\$;\\
        \\\csvautotabular{data_rates-$kk.csv}\n
        \\">>$latexfile;
done


echo -e "\\\end{document}" >> $latexfile;

pdflatex $latexfile
