while getopts "p:t:r:i:R:I:w:e:x:y:D:E:g:o:s:O:Q:F:T:G:m:M:c:a:b:" opt; do
    case $opt in
        p) ram=$OPTARG ;;
        t) time=$OPTARG ;;
        r) chifRe=$OPTARG ;;
        i) chifIm=$OPTARG ;;
        R) chidRe=$OPTARG ;;
        I) chidIm=$OPTARG ;;
        w) wavelength=$OPTARG ;;
        e) des_region=$OPTARG ;;
        x) design_x=$OPTARG ;;
        y) design_y=$OPTARG ;;
        D) des_param0=$OPTARG ;;
        E) des_param1=$OPTARG ;;
        g) gpr=$OPTARG ;;
        o) obj=$OPTARG ;;
        s) save=$OPTARG ;;
        O) opttol=$OPTARG ;;
        Q) Qabs=$OPTARG ;;
        F) fakeSratio=$OPTARG ;;
        T) iter_period=$OPTARG ;;
        m) pml_sep=$OPTARG ;;
        M) pml_thick=$OPTARG ;;
        c) do_checks=$OPTARG ;;
        a) nprojx=$OPTARG ;;
        b) nprojy=$OPTARG ;;
        :) echo "Option -$OPTARG requires an argument." >&2 ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${D}_${V}    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --signal=B:TERM@300
#SBATCH --mem=${ram}mb                     # Job memory request
#SBATCH --time=$time:10:00               # Time limit hours
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.out
module purge
module load anaconda3/2023.9
conda activate phd

python run_parallel_cloak_bound_1side.py \
    -chifRe $chifRe \
    -chifIm $chifIm \
    -chidRe $chidRe \
    -chidIm $chidIm \
    -wavelength $wavelength \
    -des_region $des_region \
    -design_x $design_x \
    -design_y $design_y \
    -des_param0 $des_param0 \
    -des_param1 $des_param1 \
    -gpr $gpr \
    -obj $obj \
    -save $save \
    -opttol $opttol \
    -Qabs $Qabs \
    -fakeSratio $fakeSratio \
    -iter_period $iter_period \
    -pml_sep $pml_sep \
    -pml_thick $pml_thick \
    -nprojx $nprojx \
    -nprojy $nprojy
exit 0
EOT