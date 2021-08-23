# $origin is loaded in the runseries.sh script that calls directories.sh, and corresponds to the
# path of the folder from where this runseries.sh will be executed

# Source of all the project
root="/path/to/HHO-pacs"

# Location for the schemes' executable
# Change it to redirect to personale tests

executable=$root"/build/Test-aurelio/$executable_name"
old_version_executable=$root"/cmake-build-default/Schemes/$old_version_executable_name"
#executable=$root"/build/Schemes/$executable_name"

# Location of mesh files
meshdir=$root"/meshes"

# Location for all outputs.
outdir=$origin"/outputs"

# Names for error file and latex file
errorsfile_new="data_rates_new.dat"
errorsfile_old="data_rates_old.dat"
errorsfile="data_rates.dat"
latexfile="rates.tex"
csvfile="data_rates.csv"
