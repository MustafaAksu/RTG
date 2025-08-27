# rtg_cli_runbook.ps1
# PowerShell runbook for RTG experiments.
# Usage: dot source it ->  . .\rtg_cli_runbook.ps1    then call the functions below.
# Requires: PowerShell 7+, Python env with rtg_kernel_rg_v3.py on PATH.

# --- Optional: target Intel GPU with DPNP/oneAPI ---
# Uncomment the next line if you want to force Level Zero GPU
# $env:SYCL_DEVICE_FILTER = "level_zero:gpu"

# Quick check of active SYCL device (optional):
function Show-SYCLDevice {
    python - << 'PY'
import sys
try:
    import dpctl
    print("SYCL device:", dpctl.SyclQueue().device)
except Exception as e:
    print("dpctl not available or no SYCL device:", e, file=sys.stderr)
PY
}

# Common t-grids
$Tfine = "0.25,0.2,0.15,0.1,0.07,0.05,0.035,0.025,0.02,0.015,0.01,0.007,0.005"
$Tstd  = "0.25,0.2,0.15,0.1,0.07,0.05,0.035,0.025,0.02,0.015,0.01"

# 1) Ultrafine parameter scan (same spin)
function Scan-Ultrafine {
    param(
        [string]$Attrs = "attrs_2048.npz",
        [string]$Report = "scan_2048_ultrafine.json",
        [string]$RhoGrid = "1.16,1.18,1.20,1.22,1.24",
        [string]$A2Grid  = "0.45,0.50,0.55",
        [string]$BetaGrid= "1.35,1.40,1.45,1.50,1.55,1.60,1.65"
    )
    python rtg_kernel_rg_v3.py scan --attrs $Attrs --report $Report `
        --rho-scale-grid $RhoGrid --a2-grid $A2Grid --beta-grid $BetaGrid `
        --spin-modes same --t-grid $Tstd --mds-dim 4 --max-nodes-eig 1200
}

# 2) Top-N from scan (sorted by |sdim| then stress)
function Scan-Top {
    param(
        [string]$Report = "scan_2048_ultrafine.json",
        [int]$TopN = 20,
        [string]$OutCsv = "scan_2048_ultrafine_top20.csv"
    )
    $d = Get-Content $Report | ConvertFrom-Json
    $d.scan | Sort-Object @{Expression={ [math]::Abs($_.spectral_dim) }}, mds_stress | `
        Select-Object -First $TopN | Export-Csv $OutCsv -NoTypeInformation
    Write-Host "wrote $OutCsv"
}

# 3) Build + analyze one kernel (same spin)
function BuildAnalyze-One {
    param(
        [string]$Attrs = "attrs_2048.npz",
        [double]$RhoScale = 1.16,
        [double]$A2 = 0.55,
        [double]$Beta = 1.40,
        [string]$Spin = "same",
        [int]$MaxEig = 1200,
        [int]$Bootstrap = 16,
        [int]$Seed = 71
    )
    $bp = $Beta.ToString("0.00").Replace('.', 'p')
    $a2s = $A2.ToString("0.00").Replace('.', 'p')
    $rs = $RhoScale.ToString("0.00").Replace('.', 'p')
    $k = "K_$(Split-Path $Attrs -LeafBase)_${Spin}_rho${rs}_a2${a2s}_b${bp}.npy"
    python rtg_kernel_rg_v3.py build-kernel --attrs $Attrs --spin-mode $Spin `
        --a2 $A2 --beta $Beta --rho-scale $RhoScale --out $k --plots
    $rep = $k -replace '\.npy$','_tfine.json'
    python rtg_kernel_rg_v3.py analyze --kernel $k --report $rep `
        --t-grid $Tfine --bootstrap $Bootstrap --seed $Seed `
        --mds-dim 4 --max-nodes-eig $MaxEig --plots
}

# 4) Beta sweep around a choice (same spin)
function Sweep-Beta {
    param(
        [string]$Attrs = "attrs_2048.npz",
        [double]$RhoScale = 1.16,
        [double]$A2 = 0.55,
        [double[]]$Betas = @(1.35,1.40,1.45,1.50,1.55,1.60,1.65),
        [int]$Bootstrap = 16, [int]$Seed = 71
    )
    foreach ($b in $Betas) {
        BuildAnalyze-One -Attrs $Attrs -RhoScale $RhoScale -A2 $A2 -Beta $b -Bootstrap $Bootstrap -Seed $Seed
    }
    # Collate
    $files = Get-ChildItem ("K_*_same_rho{0}_a2{1}_b*_tfine.json" -f ($RhoScale.ToString("0.00").Replace('.','p')), ($A2.ToString("0.00").Replace('.','p')))
    $rows = foreach ($f in $files) {
        $r = Get-Content $f.FullName | ConvertFrom-Json
        if ($r.kernel_file -match '_b(\d+)p(\d+)\.npy$') { $beta = [double]("{0}.{1}" -f $matches[1], $matches[2]) } else { $beta = $null }
        [pscustomobject]@{ beta=$beta; sdim=$r.spectral_dimension_mean; se=$r.spectral_dimension_se; mds=$r.mds_stress }
    }
    $rows | Sort-Object beta | Export-Csv ("beta_sweep_rho{0}_a2{1}.csv" -f ($RhoScale.ToString("0.00").Replace('.','p')), ($A2.ToString("0.00").Replace('.','p'))) -NoTypeInformation
}

# 5) Phase-encode-opp variants (beta ignored)
function Analyze-Opp {
    param(
        [string]$Attrs = "attrs_2048.npz",
        [double[]]$RhoScales = @(1.16,1.20),
        [double[]]$A2s = @(0.45,0.50,0.55),
        [int]$Bootstrap = 16, [int]$Seed = 71
    )
    foreach ($rs in $RhoScales) {
        foreach ($a2 in $A2s) {
            $rstr = $rs.ToString("0.00").Replace('.','p')
            $a2s  = $a2.ToString("0.00").Replace('.','p')
            $k = "K_$(Split-Path $Attrs -LeafBase)_opp_rho${rstr}_a2${a2s}_bIGNORED.npy"
            python rtg_kernel_rg_v3.py build-kernel --attrs $Attrs --spin-mode phase-encode-opp `
                --a2 $a2 --beta 0.0 --rho-scale $rs --out $k --plots
            $rep = $k -replace '\.npy$','_tfine.json'
            python rtg_kernel_rg_v3.py analyze --kernel $k --report $rep `
                --t-grid $Tfine --bootstrap $Bootstrap --seed $Seed `
                --mds-dim 4 --max-nodes-eig 1200 --plots
        }
    }
}

# 6) RG windows
function Run-RG {
    param(
        [string]$Attrs = "attrs_2048.npz",
        [string]$Report = "rg_2048_ultrafine.json",
        [string]$RhoGrid = "1.15,1.20,1.25,1.30,1.35,1.40",
        [string]$BetaGrid= "1.30,1.40,1.50,1.60",
        [int]$Steps = 5
    )
    python rtg_kernel_rg_v3.py rg --attrs $Attrs --steps $Steps --report $Report `
        --radius-scale 1.0 --rho-scale-grid $RhoGrid --beta-grid $BetaGrid `
        --t-grid $Tstd --mds-dim 4 --max-nodes-eig 1200 --seed 13
}

# 7) Size-scaling at fixed parameters (same spin)
function Size-Scaling {
    param(
        [int[]]$Ns = @(1024,2048,4096,8192),
        [double]$RhoScale = 1.20, [double]$A2 = 0.50, [double]$Beta = 1.60
    )
    foreach ($n in $Ns) {
        $attrs = "attrs_${n}.npz"
        if (!(Test-Path $attrs)) { python rtg_kernel_rg_v3.py synth --n $n --seed 1 --out $attrs }
        BuildAnalyze-One -Attrs $attrs -RhoScale $RhoScale -A2 $A2 -Beta $Beta -Seed 71 -Bootstrap 12 -MaxEig 1600
    }
    # Collate to CSV
    $files = Get-ChildItem ("K_*_same_rho{0}_a2{1}_b{2}_tfine.json" -f ($RhoScale.ToString("0.00").Replace('.','p')), ($A2.ToString("0.00").Replace('.','p')), ($Beta.ToString("0.00").Replace('.','p')))
    $rows = foreach ($f in $files) {
        $r = Get-Content $f.FullName | ConvertFrom-Json
        if ($r.kernel_file -match '^K_(\d+)_same_') { $n = [int]$matches[1] } else { $n = $null }
        [pscustomobject]@{ n=$n; spectral_dimension_mean=$r.spectral_dimension_mean; spectral_dimension_se=$r.spectral_dimension_se; mds_stress=$r.mds_stress }
    }
    $rows | Sort-Object n | Export-Csv ("n_scaling_summary.csv") -NoTypeInformation
}

# 8) Multi-seed reproducibility for one kernel
function MultiSeed-One {
    param(
        [string]$Kernel = "K_2048_same_rho1p16_a2p55_b1p40.npy",
        [int[]]$Seeds = @(41,42,43,44,45,73,97),
        [int]$Bootstrap = 12
    )
    foreach ($s in $Seeds) {
        python rtg_kernel_rg_v3.py analyze --kernel $Kernel `
            --report ("{0}_seed{1}.json" -f ($Kernel -replace '\.npy$',''), $s) `
            --t-grid $Tfine --bootstrap $Bootstrap --seed $s --mds-dim 4 --max-nodes-eig 1200
    }
}
