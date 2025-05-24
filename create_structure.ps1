# Script pour créer la structure de dossiers de NeuroLite

# Dossier racine
$root = "neuro_lite"

# Création des dossiers principaux
$mainDirs = @(
    "core",
    "architectures",
    "modules",
    "optimization",
    "symbolic",
    "training",
    "utils",
    "deploy",
    "experimental"
)

# Création des sous-dossiers
$subDirs = @{
    "architectures" = @("mlp_mixer", "hyper_mixer", "state_space")
    "modules" = @("attention", "memory", "projection", "routing")
    "optimization" = @("distillation")
    "training" = @("trainers", "callbacks", "losses", "metrics")
    "deploy" = @("onnx", "tensorrt", "mobile")
    "experimental" = @("federated", "automl", "self_supervised")
}

# Fonction pour créer un dossier s'il n'existe pas
function Ensure-Directory {
    param([string]$path)
    if (-not (Test-Path -Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "Créé: $path"
    } else {
        Write-Host "Existe déjà: $path"
    }
}

# Création de la structure
Write-Host "Création de la structure de dossiers..." -ForegroundColor Cyan

# Créer le dossier racine
Ensure-Directory $root

# Créer les dossiers principaux
foreach ($dir in $mainDirs) {
    $path = Join-Path $root $dir
    Ensure-Directory $path
}

# Créer les sous-dossiers
foreach ($parent in $subDirs.Keys) {
    foreach ($child in $subDirs[$parent]) {
        $path = Join-Path $root $parent $child
        Ensure-Directory $path
    }
}

# Créer les fichiers __init__.py dans chaque dossier Python
Write-Host "`nCréation des fichiers __init__.py..." -ForegroundColor Cyan

function Create-InitFiles {
    param([string]$basePath)
    
    Get-ChildItem -Path $basePath -Directory -Recurse | ForEach-Object {
        $initFile = Join-Path $_.FullName "__init__.py"
        if (-not (Test-Path $initFile)) {
            "# $($_.Name) module\n" | Out-File -FilePath $initFile -Encoding utf8
            Write-Host "Créé: $initFile"
        } else {
            Write-Host "Existe déjà: $initFile"
        }
    }
}

Create-InitFiles $root

Write-Host "`nStructure créée avec succès!" -ForegroundColor Green
